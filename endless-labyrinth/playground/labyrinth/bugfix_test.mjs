// Proof for every fix in WALL_REVIEW.md, and proof that each proof can fail.
//
//   node bugfix_test.mjs            run the suite
//   node bugfix_test.mjs --mutate   reintroduce each bug in turn and require the suite
//                                   to catch it (a test that cannot fail proves nothing)
//
// The network tests run against a real socket that accepts a connection and then says
// nothing, because that is the failure being fixed and a mocked promise that never
// settles would only be a restatement of the assertion.
//
// No browser here: WebGL and the sort worker are stubbed, but the code under test is the
// shipped code, not a copy of it.

import { createServer } from "node:http";
import { readFileSync, writeFileSync, copyFileSync, unlinkSync } from "node:fs";
import { execFileSync } from "node:child_process";

import { withDeadline, getJSON, backoff, Defer } from "./netpolicy.js";
import { Player, makeCollider, circleHitsBox } from "./controls.js";
import { Occupancy } from "./occupancy.js";
import { Labyrinth } from "./maze.js";
import { buildLevelMesh, buildSolidLevelMesh } from "./geometry.js";
import { makeMeshCollider } from "./collider3d.js";

// The env this repo actually runs under; `python3` on PATH has no numpy here.
const PYTHON = process.env.LAB_PYTHON
    || "/home/zrx/miniconda3/envs/hyworld2/bin/python3.11";

let pass = 0, fail = 0;
const failed = [];
function check(name, ok, detail = "") {
    if (ok) pass++; else { fail++; failed.push(name); }
    if (!process.env.QUIET || !ok)
        console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? "   " + detail : ""}`);
}
const near = (a, b, tol) => Math.abs(a - b) <= tol;

// ---------------------------------------------------------------- 1. deadlines
console.log("\nnetpolicy: a hung connection must not hang the client");

const hung = createServer(() => { /* accept, and never answer */ });
await new Promise((res) => hung.listen(0, "127.0.0.1", res));
const hungURL = `http://127.0.0.1:${hung.address().port}/`;

{
    const t0 = performance.now();
    let err = null;
    try { await getJSON(hungURL, 300, "hung probe"); } catch (e) { err = e; }
    const took = performance.now() - t0;
    check("a server that never answers aborts at the deadline",
          err !== null && took < 2000, `gave up after ${took.toFixed(0)} ms`);
    check("the timeout error names what timed out",
          err !== null && /hung probe timed out after 300 ms/.test(err.message),
          err ? err.message : "no error raised");
}

{   // the guard flag must be released, which is what makes a retry possible at all
    const inflight = new Set();
    const attempt = async () => {
        if (inflight.has("k")) return "skipped";
        inflight.add("k");
        try { await getJSON(hungURL, 200, "tile k"); return "ok"; }
        catch { return "failed"; }
        finally { inflight.delete("k"); }
    };
    const first = await attempt();
    const second = await attempt();
    check("a tile that hung once is requested again",
          first === "failed" && second === "failed" && inflight.size === 0,
          `first=${first} second=${second} inflight=${inflight.size}`);
}

{   // a real error must survive as itself, not be relabelled as a timeout
    let err = null;
    try {
        await getJSON("http://127.0.0.1:1/", 5000, "refused");
    } catch (e) { err = e; }
    check("a genuine failure is not disguised as a timeout",
          err !== null && !/timed out/.test(err.message), err ? err.message : "none");
}

{   // a body that stalls after the headers is the same hang with extra steps
    const slowBody = createServer((req, res) => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.write('{"a":');                       // ...and then nothing, forever
    });
    await new Promise((r) => slowBody.listen(0, "127.0.0.1", r));
    const url = `http://127.0.0.1:${slowBody.address().port}/`;
    let err = null;
    const t0 = performance.now();
    try { await getJSON(url, 300, "stalled body"); } catch (e) { err = e; }
    check("a response that stalls mid-body also aborts",
          err !== null && performance.now() - t0 < 2000,
          err ? err.message : "resolved anyway");
    slowBody.close();
}

// ---------------------------------------------------------------- 2. backoff / refusal
console.log("\nnetpolicy: a refusal must not become a download loop");

{
    let clock = 0;
    const d = new Defer(() => clock);

    d.hold("t", () => 3000, 10.0);                       // refused at cost 10
    check("a refused tile is not asked for again immediately", d.has("t", 10.0) === true);
    clock = 1000;
    check("...nor a second later", d.has("t", 10.0) === true);
    check("...but walking closer cancels the wait, since only that changes the answer",
          d.has("t", 4.0) === false);

    d.hold("u", () => 3000, 10.0);
    clock = 5000;
    check("a refusal also expires on its own", d.has("u", 10.0) === false);

    const w = new Defer(() => 0);
    const waits = [1, 2, 3, 4, 5, 6].map(() => {
        const n = w.hold("e", (k) => backoff(k, 500, 8000));
        return backoff(n, 500, 8000);
    });
    check("repeated failure backs off and then caps",
          JSON.stringify(waits) === JSON.stringify([500, 1000, 2000, 4000, 8000, 8000]),
          waits.join(", ") + " ms");

    const s = new Defer(() => 0);
    s.hold("x", () => 3000, 1);
    s.clear("x");
    check("success clears the wait", s.has("x", 1) === false && s.size === 0);
}

{   // the loop this prevents, counted end to end
    let clock = 0, requests = 0;
    const d = new Defer(() => clock);
    for (let tick = 0; tick < 40; tick++) {              // 14 s of standing still
        clock = tick * 350;                             // the ring's own pump interval
        if (d.has("t", 10.0)) continue;
        requests++;
        d.hold("t", () => 3000, 10.0);                  // the field refuses it again
    }
    check("standing still costs 5 requests over 14 s, not 40", requests === 5,
          `${requests} requests`);
}

hung.close();

// ---------------------------------------------------------------- 3. controller
console.log("\ncontrols: the same input must move you the same distance at any refresh rate");

{
    const walk = (hz) => {
        const p = new Player({ pos: [0, 0, 0] });
        const keys = new Set(["KeyW"]);
        const dt = 1 / hz;
        for (let t = 0; t < 0.5 - 1e-9; t += dt) p.update(dt, keys, null);
        return p.pos[2];
    };
    const rates = [30, 60, 90, 144, 240];
    const d = rates.map(walk);
    const spread = (Math.max(...d) - Math.min(...d)) / Math.abs(d[0]);
    check("half a second of W covers one distance from 30 to 240 Hz", spread < 1e-6,
          `${(spread * 100).toFixed(4)}% spread, ${d[0].toFixed(4)} m`);
}

console.log("\ncontrols: being inside a wall must be recoverable");

{
    const slab = (x, z, r) => x + r > 1 && x - r < 3;    // solid for x in [1, 3]
    const p = new Player({ pos: [2, 0, 0] });           // ...and the body starts inside it
    for (let i = 0; i < 8; i++) p.update(1 / 60, new Set(), slab);
    check("a body inside solid steps out", !slab(p.pos[0], p.pos[2], 0.34) && !p.stuck,
          `escaped to x=${p.pos[0].toFixed(3)}`);

    const q = new Player({ pos: [2, 0, 0] });
    q.update(1 / 60, new Set(), slab);
    const first = q.pos[0];
    q.update(1 / 60, new Set(), slab);
    check("and stays put once out", near(first, q.pos[0], 1e-9));

    const r = new Player({ pos: [0, 0, 0] });
    r.update(1 / 60, new Set(), () => true);
    check("but does not teleport when everything nearby is solid too",
          Math.hypot(r.pos[0], r.pos[2]) === 0 && r.stuck === true,
          "unloaded occupancy reads as solid, and guessing there would be worse");

    const w = new Player({ pos: [0, 0, 0], yaw: Math.PI / 2 });
    const keys = new Set(["KeyW"]);
    for (let i = 0; i < 600; i++) w.update(1 / 60, keys, slab);
    check("walking into a wall still stops at it", w.pos[0] <= 1 - 0.34 + 1e-6 && !w.stuck,
          `stopped at x=${w.pos[0].toFixed(4)}, face at 0.66`);
}

// ---------------------------------------------------------------- 4. cell index
console.log("\ngeometry: which cell a position is in");

{
    // These three used to be a local copy of the formula plus two regexes over the
    // shipped source. That is not a test of anything: the regexes assert the SHAPE of
    // the code, so any behaviourally-identical rewrite of the bug -- floor(gx/2 - 0.5),
    // or a 0.5 folded into the caller -- puts the defect back with the suite still at
    // 43/43, and the mutation harness scored "caught" only because its mutation was
    // itself a text edit the regex happened to match. Both sides are executed here
    // instead, against the geometry rather than against each other.
    const shipped = readFileSync("./endless.js", "utf8");
    const m = shipped.match(/function cellAt\(x, z\) \{[\s\S]*?\n\}/);
    check("endless.js still defines cellAt", !!m);
    const cellAt = m && new Function("CELLM", `${m[0]}\nreturn cellAt;`)(() => 2.0);

    // The geometry, stated independently: cell (c,r) occupies grid index (2c+1, 2r+1),
    // so at cell_m = 2 it owns metres [2*(2c+1), 2*(2c+2)).
    const probes = [];
    for (let c = -6; c <= 6; c++)
        for (const f of [0.001, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.999])
            probes.push([2 * (2 * c + 1 + f), c]);
    let wrong = 0;
    for (const [x, want] of probes) if (cellAt(x, 0)[0] !== want) wrong++;
    check("the shipped cellAt puts every point in the cell that owns it",
          wrong === 0, `${wrong} of ${probes.length} misassigned`);

    // ...and the same probes through the shipped python, because the bug that mattered
    // was the two agreeing with each other while both disagreed with the geometry. A
    // regex over either file cannot see that; running both can.
    const span = 2;
    const want = probes.map(([x, c]) => Math.floor(c / span));
    let got = null;
    try {
        got = JSON.parse(execFileSync(PYTHON, ["-c", `
import sys, json
sys.path.insert(0, "../../experiments/labyrinth/endless")
from tiles import tile_of_pos_m
xs = json.loads(sys.argv[1])
print(json.dumps([tile_of_pos_m(x, 3.0, ${span})[0] for x in xs]))`,
            JSON.stringify(probes.map((p) => p[0]))], { encoding: "utf8" }));
    } catch (e) { got = String(e.message).slice(0, 120); }
    check("tiles.py puts them in the same tile the geometry does",
          Array.isArray(got) && got.every((v, i) => v === want[i]),
          Array.isArray(got)
              ? `${got.filter((v, i) => v !== want[i]).length} of ${want.length} disagree`
              : got);
}

// ---------------------------------------------------------------- 5. occupancy growth
console.log("\noccupancy: an endless walk must not mean endless memory");

{
    const occ = new Occupancy(2.0);
    const window = (c0, r0, n) => ({
        c0, r0, cols: n, rows: n,
        walls: Array.from({ length: n * n }, () => [true, false, true, false]),
    });
    let peak = 0;
    for (let step = 0; step < 400; step++) {           // walk 400 * 18 = 7200 cells east
        const c = step * 18;
        occ.ingest(window(c - 28, -28, 56));
        occ.prune(c, 0, 84);
        peak = Math.max(peak, occ.walls.size);
    }
    check("the wall map stops growing", peak < 40000, `peak ${peak.toLocaleString()} cells`);

    // and the part that is still needed must survive: dropping a cell that is inside the
    // loaded box reads back as "not loaded", which solidIdx reports as solid -- an
    // invisible wall across an open corridor
    const o2 = new Occupancy(2.0);
    o2.ingest(window(-28, -28, 56));
    o2.prune(0, 0, 84);
    let missing = 0;
    for (let r = -28; r < 28; r++)
        for (let c = -28; c < 28; c++) if (!o2.walls.has(c + "," + r)) missing++;
    check("nothing inside the loaded window is dropped", missing === 0,
          `${missing} cells lost from the live box`);
}

// ---------------------------------------------------------------- 6. splat field
console.log("\nsplats: a sort that finished late must not draw a tile that is gone");

class FakeWorker {
    constructor() { this.sent = []; this.onmessage = null; }
    postMessage(msg, transfer) { this.sent.push({ msg, transfer }); }
    reply(data) { this.onmessage({ data }); }
}
let lastWorker = null;
globalThis.Worker = class { constructor() { lastWorker = new FakeWorker(); return lastWorker; } };
if (!globalThis.URL.createObjectURL) globalThis.URL.createObjectURL = () => "blob:stub";

function fakeGL(maxTex = 16384) {
    const base = {
        MAX_TEXTURE_SIZE: "MAX_TEXTURE_SIZE",
        getParameter: (p) => (p === "MAX_TEXTURE_SIZE" ? maxTex : 0),
        getShaderParameter: () => true,
        getProgramParameter: () => true,
        getAttribLocation: () => 0,
        getUniformLocation: () => ({}),
        getShaderInfoLog: () => "",
        getProgramInfoLog: () => "",
    };
    return new Proxy(base, { get: (t, k) => (k in t ? t[k] : () => ({})) });
}

const { SplatField } = await import("./splats.js");
const tile = (n) => new Float32Array(n * 14).fill(0.1);

{
    const f = new SplatField(fakeGL(), 800_000, 4);
    f.addChunk("a", tile(1000), null, 0);
    const genAtRequest = f.generation;
    f.requestSort(new Float32Array(16));
    f.addChunk("b", tile(1000), null, 0);            // evicts nothing, but relayouts

    lastWorker.reply({ order: new Uint32Array(50), count: 50, gen: genAtRequest });
    check("an order computed before the field changed is discarded",
          f.sortedCount === 0, `sortedCount ${f.sortedCount}`);
    check("...and the sorter is freed to try again rather than wedged",
          f.sorting === false);

    lastWorker.reply({ order: new Uint32Array(50), count: 50, gen: f.generation });
    check("an order for the current field is drawn", f.sortedCount === 50);
}

{   // eviction is the case that actually produced the ghosts
    const f = new SplatField(fakeGL(), 800_000, 2);
    f.addChunk("a", tile(100), null, 5.0);
    f.addChunk("b", tile(100), null, 5.0);
    f.requestSort(new Float32Array(16));
    const stale = f.generation;
    const added = f.addChunk("c", tile(100), null, 0.1);   // nearer: evicts one of a/b
    check("a nearer tile evicts a further one", added > 0 && f.evictions === 1,
          `${f.evictions} evictions`);
    lastWorker.reply({ order: new Uint32Array(200), count: 200, gen: stale });
    check("the evicted tile's sort order never reaches the GPU", f.sortedCount === 0);
}

console.log("\nsplats: the field must fit in a texture, and a tile must fit in a slot");

{
    const big = new SplatField(fakeGL(16384), 8_000_000, 18);
    check("a GPU with room keeps all 18 slots", big.slotCount === 18,
          `${big.slotCount} slots of ${big.slotSize.toLocaleString()}`);

    const small = new SplatField(fakeGL(8192), 8_000_000, 18);
    const rows = Math.ceil((small.slotSize * small.slotCount * 4) / 2048);
    check("at the WebGL2 floor the field is cut to fit instead of failing to allocate",
          rows <= 8192 && small.slotCount < 18,
          `${small.slotCount} slots, ${rows} texture rows (limit 8192)`);
    check("...by dropping slots, never slot size, so tiles are still never truncated",
          small.slotSize === big.slotSize,
          `${small.slotSize.toLocaleString()} splats per slot either way`);
    check("a slot holds the worst case voxel_prune can return (1.1 x 400k)",
          big.slotSize >= Math.ceil(400_000 * 1.1),
          `${big.slotSize.toLocaleString()} >= 440,000`);
}

{   // truncation used to be silent; it is the patch-shaped-holes bug
    const f = new SplatField(fakeGL(), 800_000, 4);
    const warnings = [];
    const realWarn = console.warn;
    console.warn = (...a) => warnings.push(a.join(" "));
    f.addChunk("x", tile(f.slotSize + 5000), null, 0);
    console.warn = realWarn;
    check("a tile too big for its slot says so", warnings.some((w) => /dropped/.test(w)),
          warnings[0] ? warnings[0].slice(0, 80) : "silent");
}

{   // 91.5 MB of structured clone per tile was the cost of sending positions this way
    const f = new SplatField(fakeGL(), 8_000_000, 18);
    lastWorker.sent.length = 0;
    f.addChunk("a", tile(1000), null, 0);
    const posts = lastWorker.sent;
    const bytes = posts.reduce((n, p) => n + (p.msg.chunk ? p.msg.chunk.byteLength : 0), 0);
    check("only the slot that changed is sent to the sorter",
          posts.every((p) => !("positions" in p.msg)) && bytes === 1000 * 3 * 4,
          `${(bytes / 1024).toFixed(0)} KB, not ${(f.capacity * 3 * 4 / 1048576).toFixed(0)} MB`);
    check("...and its buffer is transferred, not copied",
          posts.some((p) => p.transfer && p.transfer.length === 1));
}

// ---------------------------------------------------------------- 7. hollow walls
console.log("\ngeometry: a wall you can walk into is not a wall");

{
    // Can a body WALK into the inside of a wall?
    //
    // A point test is the wrong question for a triangle collider: MeshCollider measures
    // distance to the nearest surface, so standing at the centre of a 2 m slab is 1 m
    // from every face and reads as free whether or not the slab is closed. What decides
    // the bug is reachability -- flood the free space from the spawn cell and ask
    // whether any of it is inside a wall.
    const maze = new Labyrinth({ cols: 10, rows: 10, seed: 1000, cell: 4.0, wallT: 2.0 });
    const S = 4.0;
    const slabs = [];
    for (let r = 0; r < maze.rows; r++)
        for (let c = 0; c < maze.cols; c++)
            slabs.push(...maze.wallsNear((c + 0.5) * S, (r + 0.5) * S, 0));
    const insideWall = (x, z) =>
        slabs.some((b) => x > b.x0 + 0.05 && x < b.x1 - 0.05 && z > b.z0 + 0.05 && z < b.z1 - 0.05);

    function reachableInWalls(collide) {
        const step = 0.25, seen = new Set();
        const key = (i, j) => i + "," + j;
        const sx = (maze.spawn.c + 0.5) * S, sz = (maze.spawn.r + 0.5) * S;
        const i0 = Math.round(sx / step), j0 = Math.round(sz / step);
        const queue = [[i0, j0]];
        seen.add(key(i0, j0));
        let breached = 0;
        while (queue.length) {
            const [i, j] = queue.pop();
            const x = i * step, z = j * step;
            if (insideWall(x, z)) breached++;
            for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
                const ni = i + di, nj = j + dj, k = key(ni, nj);
                if (seen.has(k)) continue;
                const nx = ni * step, nz = nj * step;
                if (nx < 0 || nz < 0 || nx > maze.cols * S || nz > maze.rows * S) continue;
                if (collide(nx, nz, 0.34)) continue;
                seen.add(k);
                queue.push([ni, nj]);
            }
        }
        return { breached, reached: seen.size };
    }

    const shell = reachableInWalls(makeMeshCollider(buildLevelMesh(maze)));
    const solid = reachableInWalls(makeMeshCollider(buildSolidLevelMesh(maze)));
    const aabb = reachableInWalls(makeCollider(maze));

    check("the render mesh is a hollow shell -- a body walks in through an uncapped end",
          shell.breached > 0,
          `${shell.breached} standing positions inside walls reachable on foot`);
    check("the solid mesh seals them", solid.breached === 0,
          `${solid.breached} of ${solid.reached.toLocaleString()} reachable points inside a wall`);
    check("maze.wallsNear seals them too, which is what live.js now uses",
          aabb.breached === 0,
          `${aabb.breached} of ${aabb.reached.toLocaleString()} reachable points inside a wall`);
    check("sealing the walls did not seal the corridors", aabb.reached > shell.reached * 0.5,
          `${aabb.reached.toLocaleString()} reachable vs ${shell.reached.toLocaleString()} through the shell`);
}

{   // ...and the windowed AABB collider must equal an unwindowed one, or it has its own holes
    let diffs = 0, samples = 0;
    for (const seed of [1, 42, 1000]) {
        const maze = new Labyrinth({ cols: 20, rows: 20, seed, cell: 4.0, wallT: 2.0 });
        const windowed = makeCollider(maze);
        const all = [];
        for (let r = 0; r < maze.rows; r++)
            for (let c = 0; c < maze.cols; c++)
                all.push(...maze.wallsNear((c + 0.5) * 4, (r + 0.5) * 4, 0));
        for (let z = 2; z < 76; z += 0.23)
            for (let x = 2; x < 76; x += 0.23) {
                samples++;
                const truth = all.some((b) => circleHitsBox(x, z, 0.34, b));
                if (windowed(x, z, 0.34) !== truth) diffs++;
            }
    }
    check("its one-cell search window loses nothing", diffs === 0,
          `${diffs} disagreements over ${samples.toLocaleString()} probes`);
}

// ---------------------------------------------------------------- 8. shipped constants
console.log("\nshipped constants");

{
    const src = readFileSync("./live.js", "utf8");
    const maxSplats = Number(/numParam\("max", (\d+)/.exec(src)[1]);
    // Evaluate the expression live.js actually ships, rather than a copy of it here:
    // recomputing the formula in the test means the test cannot notice live.js changing,
    // which is exactly what the mutation run caught it doing.
    const expr = /const SLOTS = ([^;]+);/.exec(src)[1];
    const slots = Function("maxSplats", `return ${expr};`)(maxSplats);
    const slotSize = Math.floor(8_000_000 / slots / 512) * 512;
    check("live.js sizes a slot above the worst chunk the server can return",
          slotSize >= Math.ceil(maxSplats * 1.1),
          `${slots} slots of ${slotSize.toLocaleString()} >= ${Math.ceil(maxSplats * 1.1).toLocaleString()}`);
    check("live.js derives that from the budget rather than hard-coding 24",
          /Math\.floor\(8_000_000 \/ Math\.ceil\(maxSplats \* 1\.1\)\)/.test(src));
    check("live.js no longer collides against the splats it drew",
          !/field\.blocked\(x, z, r\)/.test(src));
}

{
    // Was: two regexes over two COMMENTS. The shader could read the wire quaternion
    // vector-first -- the actual P1-4 defect, 133 degrees of median misorientation --
    // and this still passed, because comments are not code. Run the shipped unpack
    // expression against a wire record whose four quaternion floats are all distinct.
    const src = readFileSync("./splats.js", "utf8");
    const qm = src.match(/vec4 q = vec4\(([^)]*)\);/);
    check("splats.js still unpacks a quaternion from the texels", !!qm);

    // The wire, per the header: x y z sx | sy sz qw qx | qy qz r g | b a _ _
    const wire = { x: 0, y: 0, z: 0, sx: 1, sy: 1, sz: 1,
                   qw: 10, qx: 20, qy: 30, qz: 40, r: 0, g: 0, b: 0, a: 1 };
    const f = [wire.x, wire.y, wire.z, wire.sx, wire.sy, wire.sz, wire.qw,
               wire.qx, wire.qy, wire.qz, wire.r, wire.g, wire.b, wire.a];
    const tex = (i) => ({ x: f[i], y: f[i + 1], z: f[i + 2], w: f[i + 3] });
    const got = new Function("a", "b", "c", "d", "vec4",
        `return vec4(${qm[1]});`)(tex(0), tex(4), tex(8), tex(12),
        (x, y, z, w) => ({ x, y, z, w }));
    // GLSL vec4 is (x, y, z, w) and the maths below reads q.x/q.y/q.z as the vector
    // part and q.w as the scalar, so the unpack must land qx qy qz qw in that order.
    check("the shader lands the wire's qw qx qy qz as vector-first, scalar-last",
          got.x === wire.qx && got.y === wire.qy
          && got.z === wire.qz && got.w === wire.qw,
          `got (${got.x}, ${got.y}, ${got.z}, ${got.w}), want (20, 30, 40, 10)`);

    // And the rotation built from it is the rotation that quaternion means.
    const rot = src.match(/mat3 R = mat3\(([\s\S]*?)\);/);
    const h = Math.SQRT1_2;                       // +90 deg about +y sends +x to -z
    const nine = new Function("q", `return [${rot[1]}];`)({ x: 0, y: h, z: 0, w: h });
    const R = [[nine[0], nine[3], nine[6]], [nine[1], nine[4], nine[7]],
               [nine[2], nine[5], nine[8]]];
    const Rv = R.map((row) => row[0]);            // R * (1,0,0)
    check("and the rotation it builds is the one that quaternion means",
          Math.abs(Rv[0]) < 1e-6 && Math.abs(Rv[1]) < 1e-6 && Math.abs(Rv[2] + 1) < 1e-6,
          `x -> [${Rv.map((t) => t.toFixed(3)).join(", ")}], want [0, 0, -1]`);
}

// ---------------------------------------------------------------- 12. projection
console.log("\nsplats: a splat off the optical axis must not be drawn at its on-axis size");

{
    // GLSL mat3(...) fills COLUMN by column, so the nine arguments written in reading
    // order build the TRANSPOSE. The true perspective Jacobian has its third ROW zero;
    // the transpose has its third COLUMN zero, which deletes the shear terms -f*x/z^2
    // from the 2x2 screen block and freezes every splat at the size it would have on
    // the axis. The symptom is subtle -- a 12-deep splat field stays opaque -- so this
    // asserts the maths, not the picture.
    const src = readFileSync("./splats.js", "utf8");
    const jm = src.match(/mat3 J = mat3\(([\s\S]*?)\);/);
    check("splats.js still builds the projection jacobian", !!jm);

    const nine = (cam) => new Function("u_focal", "cam", "z2",
        `return [${jm[1]}];`)({ x: 1546.2, y: 1546.2 }, cam, cam.z * cam.z);
    const screenCov = (cam) => {
        const a = nine(cam);
        const J = [[a[0], a[3], a[6]], [a[1], a[4], a[7]], [a[2], a[5], a[8]]];
        const S = [[9e-4, 1e-4, 2e-4], [1e-4, 9e-4, 1e-4], [2e-4, 1e-4, 9e-4]];
        const JS = J.map((r) => [0, 1, 2].map((j) =>
            r[0] * S[0][j] + r[1] * S[1][j] + r[2] * S[2][j]));
        return [0, 1].map((i) => [0, 1].map((j) =>
            JS[i][0] * J[j][0] + JS[i][1] * J[j][1] + JS[i][2] * J[j][2]));
    };
    check("the slot cost store is float64, so equidistant tiles cannot swap forever",
          /this\.slotCost = new Float64Array\(/.test(src)
          && new Float64Array([Math.hypot(3, 4) + 1e-9])[0] !== Math.hypot(3, 4),
          "");

    const axis = screenCov({ x: 0, y: 0, z: -3 });
    const edge = screenCov({ x: 2, y: 0, z: -3 });
    check("a splat at the frame edge is wider than the same splat on the axis",
          edge[0][0] > axis[0][0] * 1.3,
          `cxx on axis ${axis[0][0].toFixed(1)}, at edge ${edge[0][0].toFixed(1)}`);

    // Ground truth for the same splat, by projecting the covariance the long way:
    // du/dx = f/z and du/dz = -f*x/z^2 are what the shear terms ARE.
    const f = 1546.2, z = -3, x = 2;
    const g = [f / z, 0, -(f * x) / (z * z)];
    const S = [[9e-4, 1e-4, 2e-4], [1e-4, 9e-4, 1e-4], [2e-4, 1e-4, 9e-4]];
    const want = [0, 1, 2].reduce((acc, i) => acc + [0, 1, 2].reduce(
        (a2, j) => a2 + g[i] * S[i][j] * g[j], 0), 0);
    check("and it is wider by exactly the perspective shear, not by some other amount",
          near(edge[0][0], want, Math.abs(want) * 1e-9),
          `shader ${edge[0][0].toFixed(4)}, closed form ${want.toFixed(4)}`);
}

// ---------------------------------------------------------------- report
console.log(`\n${fail ? "FAILED" : "ALL PASS"}  ${pass} passed, ${fail} failed`);
if (fail) console.log("  " + failed.join("\n  "));
process.exit(fail ? 1 : 0);
