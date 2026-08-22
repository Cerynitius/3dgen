// Walk an endless labyrinth that is reconstructed around you.
//
// This is not live.js with a bigger world. live.js keeps its own copy of a FINITE maze
// (maze.js, a JS port of the old Cells generator) and asks the server for cells of it.
// There is no finite maze here to copy: the world is generated per chunk from a hash,
// so the client cannot know the layout without asking, and the server is the authority
// on both geometry and appearance.
//
//   /spec   metric constants, so placement is arithmetic and never registration
//   /maze   wall bits for a window of cells -> exact local collision, no round trip
//   /ring   which tiles surround a position, and which are ready
//   /tile   the Gaussians, tile-relative; add origin_m to place
//
// Tiles are a pure function of (seed, tile index), so revisiting one gets the same
// bytes. That is what makes eviction safe: anything dropped can be re-fetched identical.

import { Player, bindInput } from "./controls.js";
import { SplatField } from "./splats.js";
import * as m4 from "./mat4.js";
import { Occupancy } from "./occupancy.js";
import { withDeadline, getJSON, backoff, Defer } from "./netpolicy.js";

const params = new URLSearchParams(location.search);

// A query string is user input, and an unvalidated one fails silently and expensively:
// ?radius=abc used to send r=NaN, which the server rejects with 400 forever, and
// ?gain=abc used to reach the shader as a NaN uniform, which multiplies every alpha to
// NaN and paints a black screen with no error anywhere. Both looked like "the world
// stopped generating". Out-of-range and unparseable both fall back to the default.
function numParam(name, dflt, lo, hi, cast = parseFloat) {
    const raw = params.get(name);
    if (raw === null) return dflt;
    const v = cast(raw, 10);
    if (!Number.isFinite(v) || v < lo || v > hi) {
        console.warn(`?${name}=${raw} is not a number in [${lo}, ${hi}] -- using ${dflt}`);
        return dflt;
    }
    return v;
}

const SERVER = params.get("server") || `${location.protocol}//${location.hostname}:8092`;
const RADIUS = numParam("radius", 1, 0, 4, parseInt);
// gain is 1.0 because it now does nothing, and that is the point. It existed to thicken
// a field that composited see-through, and the reason it did was a bug in voxel_prune:
// pruning compensated each surviving splat's AREA for the ~10 layers it replaced but not
// its OPACITY, so walls kept a single layer's 0.223 and you could see through them. With
// that fixed, per-splat opacity is 0.861 and sweeping gain from 1.0 to 2.4 moves
// accumulated alpha by 0.002. Raising it now would only smear.
// dilate is the same quantity as gsplat's eps2d, which is how it could be swept at all.
const gain = numParam("gain", 1.0, 0.1, 8.0);
const dilate = numParam("dilate", 0.30, 0.0, 4.0);

const hud = {
    stats: document.getElementById("stats"),
    queue: document.getElementById("queue"),
    fatal: document.getElementById("fatal"),
    hint: document.getElementById("hint"),
};
const fatal = (title, msg) => {
    hud.fatal.style.display = "block";
    hud.fatal.textContent = `${title}\n\n${msg}`;
};

const canvas = document.getElementById("view");
const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
if (!gl) { fatal("WebGL2 is required", "This browser did not provide a webgl2 context."); throw new Error("no webgl2"); }

// ---------------------------------------------------------------- world constants
let SPEC = null;
const CELLM = () => SPEC.cell_m;          // metres per grid unit; a cell spans two

/** Cell centre in metres. Mirrors infinite_maze.cell_centre_m exactly. */
function cellCentre(c, r) {
    return [(2 * c + 1.5) * CELLM(), (2 * r + 1.5) * CELLM()];
}
/** Which cell a metric position is in. Mirrors tiles.tile_of_pos_m's cell step.
 *
 * The subtrahend is 1, not 1.5. Cell (c,r) occupies grid index 2c+1, i.e. gx in
 * [2c+1, 2c+2), so the inverse of that interval is floor((gx-1)/2). Subtracting the
 * CENTRE offset instead partitions on cell centres, which sends the western quarter of
 * every cell to c-1 -- 25% of all positions. Both this and tile_of_pos_m had it, so
 * client and server agreed with each other while both disagreed with the geometry, and
 * /ring centred one tile behind the player for a quarter of the corridor. That is a
 * quarter of a tile less prefetch in exactly the place you notice it: walking forward.
 */
function cellAt(x, z) {
    const gx = x / CELLM(), gz = z / CELLM();
    return [Math.floor((gx - 1) / 2), Math.floor((gz - 1) / 2)];
}

// ---------------------------------------------------------------- network
// Deadlines, backoff and the refusal cache all live in netpolicy.js, which explains
// what each of them is for and is tested without a browser. What belongs here is only
// how long each endpoint is allowed to take: a tile is a GPU forward pass and gets a
// minute and a half, everything else is a lookup and gets seconds.
const DEADLINE_MS = { spec: 10000, maze: 15000, ring: 8000, tile: 90000 };

// ---------------------------------------------------------------- local collision
//
// The server sends wall bits; the occupancy grid they imply is rebuilt in occupancy.js
// so the body collides against the same geometry the tiles were reconstructed from.
// Colliding against the splats instead sounds more honest but is not: the
// reconstruction has holes, and a hole in a wall you can see is a hole you walk through.
// endless_selftest.mjs diffs that rebuild against the server's own rasterisation.
const occ = new Occupancy(2.0);
const blocked = (x, z, r) => occ.blocked(x, z, r);

let loadedBox = null;
let mazeInflight = false;
let torchList = [];                        // world-space flame positions from /maze
let mazeFails = 0, mazeRetryAt = 0;
async function ensureMaze(c, r) {
    const HALF = 28, MARGIN = 10;
    if (loadedBox && c - loadedBox.c0 > MARGIN && loadedBox.c1 - c > MARGIN
        && r - loadedBox.r0 > MARGIN && loadedBox.r1 - r > MARGIN) return;
    if (mazeInflight || performance.now() < mazeRetryAt) return;
    mazeInflight = true;
    try {
        const c0 = c - HALF, r0 = r - HALF, n = HALF * 2;
        const d = await getJSON(`${SERVER}/maze?c0=${c0}&r0=${r0}&cols=${n}&rows=${n}`,
                                DEADLINE_MS.maze, "maze window");
        occ.ingest(d);
        if (d.torches) torchList = d.torches;   // world-space flame positions, for the shader relight
        occ.prune(c, r, HALF * 3);       // see occupancy.prune: the map is not a cache
        loadedBox = { c0, r0, c1: c0 + n, r1: r0 + n };
        mazeFails = 0;
        netNote = "";
    } catch (e) {
        mazeFails++;
        mazeRetryAt = performance.now() + backoff(mazeFails, 500, 8000);
        netNote = `maze window failed (${e.message})`;
        console.warn("maze window failed:", e.message);
    } finally {
        mazeInflight = false;
    }
}

// ---------------------------------------------------------------- tile streaming
// 18 slots, not 24. The slot has to hold the LARGEST tile the server can emit, and
// voxel_prune accepts anything within 1.1x of its target, so a 400k budget can return
// ~440k splats. At 24 slots each holds 333,312 and every tile is silently truncated --
// and truncating a voxel-pruned list is the one thing pruning was chosen to avoid,
// since it removes coverage in patches rather than uniformly. 18 slots gives 444,416
// each, above that worst case, and still keeps twice the ring resident.
// 20 slots x 400k: the worst tile measured is 352k, and a radius-2 ring wants 25
// tiles -- 18 slots guaranteed eviction churn on every step, and each eviction
// blanks a slot until the next sort (the "walls collapse" Rex saw). 20 narrows
// the gap; the texture fits (8M splats = 15,625 rows < 16,384).
const field = new SplatField(gl, 8_000_000, 20);

// --- dust motes: 700 soft warm points drifting in a 14m box that recentres on the
// player. Pure atmosphere; costs one tiny draw call.
const DUST_VS = `#version 300 es
in vec3 a_pos;
uniform mat4 u_vp;
uniform vec3 u_cam;
uniform float u_time;
out float v_a;
void main() {
    vec3 drift = vec3(u_time * 0.03, -u_time * 0.015, u_time * 0.02);
    vec3 wob = vec3(sin(u_time * 0.7 + a_pos.y * 13.7),
                    sin(u_time * 0.5 + a_pos.x * 17.3),
                    cos(u_time * 0.6 + a_pos.z * 11.1)) * 0.15;
    vec3 box = vec3(14.0, 5.0, 14.0);
    vec3 rel = mod(a_pos + drift + wob - u_cam + box * 0.5, box) - box * 0.5;
    vec3 world = u_cam + rel;
    gl_Position = u_vp * vec4(world, 1.0);
    float d = length(rel);
    v_a = 0.08 * (1.0 - smoothstep(2.0, 7.0, d));
    gl_PointSize = 2.0;
}`;
const DUST_FS = `#version 300 es
precision mediump float;
in float v_a;
out vec4 frag;
void main() {
    float r = length(gl_PointCoord - 0.5);
    if (r > 0.5) discard;
    frag = vec4(vec3(0.75, 0.70, 0.58), v_a * (1.0 - r * 2.0));
}`;
function dustCompile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error("dust: " + gl.getShaderInfoLog(s));
    return s;
}
const dustProg = gl.createProgram();
gl.attachShader(dustProg, dustCompile(gl.VERTEX_SHADER, DUST_VS));
gl.attachShader(dustProg, dustCompile(gl.FRAGMENT_SHADER, DUST_FS));
gl.linkProgram(dustProg);
const dustU = {};
for (const n of ["u_vp", "u_cam", "u_time"]) dustU[n] = gl.getUniformLocation(dustProg, n);
const DUST_N = 700;
const dustBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, dustBuf);
const dustPts = new Float32Array(DUST_N * 3);
for (let i = 0; i < DUST_N * 3; i++) dustPts[i] = Math.random() * (i % 3 === 1 ? 5 : 14);
gl.bufferData(gl.ARRAY_BUFFER, dustPts, gl.STATIC_DRAW);
const dustVAO = gl.createVertexArray();
gl.bindVertexArray(dustVAO);
gl.bindBuffer(gl.ARRAY_BUFFER, dustBuf);
const dustLoc = gl.getAttribLocation(dustProg, "a_pos");
gl.enableVertexAttribArray(dustLoc);
gl.vertexAttribPointer(dustLoc, 3, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);
function drawDust(view, proj, camPos, nowSec) {
    gl.useProgram(dustProg);
    gl.bindVertexArray(dustVAO);
    const vp = m4.multiply(proj, view);
    gl.uniformMatrix4fv(dustU["u_vp"], false, vp);
    gl.uniform3f(dustU["u_cam"], camPos[0], camPos[1], camPos[2]);
    gl.uniform1f(dustU["u_time"], nowSec);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.depthMask(false);
    gl.drawArrays(gl.POINTS, 0, DUST_N);
    gl.depthMask(true);
    gl.bindVertexArray(null);
}
const inflight = new Set();
const haveTile = new Set();
let served = 0, bytes = 0, lastMs = 0, queued = 0, netNote = "";

// Three different outcomes used to share one behaviour -- come back at the ring's
// 0.35 s tick, forever, at full rate:
//
//   202          the tile is still being built. Expected, but polling a 1.8 s build at
//                2.9 Hz is five wasted round trips per tile.
//   5xx/network  the server is already in trouble and is now being hit three times a
//                second per tile by every client at once.
//   added === 0  the field refused the tile because every resident one is nearer. The
//                multi-megabyte body had already been downloaded and was thrown away,
//                and would be downloaded and thrown away again 0.35 s later, forever.
//                With RADIUS >= 2 the ring asks for 25 tiles into 18 slots, so this is
//                not a corner case; it is the steady state.
//
// So each outcome names when it may be asked for again. A refusal also records the cost
// it was refused at, because the only thing that can change that answer is the player
// walking closer -- so closing the distance cancels the wait, and a timer alone does not
// have to be tuned to guess it.
const defer = new Defer();

function tileCost(origin) {
    return Math.hypot(origin[0] - player.pos[0], origin[1] - player.pos[2]);
}

async function fetchTile(tc, tr, origin) {
    const key = tc + "," + tr;
    if (inflight.has(key) || haveTile.has(key) || defer.has(key, tileCost(origin))) return;
    inflight.add(key);
    const t0 = performance.now();
    try {
        // wait=0: never block the server on a tile that is still queued -- the ring
        // poll will come back for it. Blocking here stalls every other request behind it.
        // v= busts the year-long immutable cache when the world itself changes
        const url = `${SERVER}/tile?tc=${tc}&tr=${tr}&wait=0&v=${SPEC.version}`;
        const got = await withDeadline(DEADLINE_MS.tile, `tile ${key}`, async (signal) => {
            const res = await fetch(url, { signal });
            if (res.status === 202) return { building: true };
            if (!res.ok) throw new Error("HTTP " + res.status);
            const meta = JSON.parse(res.headers.get("X-Tile-Meta") || "{}");
            return { meta, buf: await res.arrayBuffer() };
        });
        // still building: back off toward the ~1.8 s a tile actually takes, instead of
        // asking 2.9 times a second for the whole of it
        if (got.building) { defer.hold(key, (n) => Math.min(350 * n, 2000)); return; }
        const at = got.meta.origin_m || origin;
        if (got.buf.byteLength === 0) {
            // solid ground: the server answered "nothing here" -- that's a final
            // answer, not a failure. Mark resident so we never ask again.
            haveTile.add(key);
            return;
        }
        const added = field.addChunk(key, new Float32Array(got.buf),
                                     m4.translation(at[0], 0, at[1]), tileCost(at));
        if (added === 0) {                              // field full of nearer tiles
            defer.hold(key, () => 3000, tileCost(at));
            return;
        }
        defer.clear(key);
        haveTile.add(key);
        served++; bytes += got.buf.byteLength; lastMs = performance.now() - t0;
        netNote = "";
        console.log(`tile ${key}: ${added.toLocaleString()} splats, ` +
                    `${(got.buf.byteLength / 1048576).toFixed(1)}MB in ${lastMs.toFixed(0)}ms`,
                    got.meta.timing);
    } catch (e) {
        defer.hold(key, (n) => backoff(n, 500, 8000));
        netNote = `tile ${key} failed (${e.message})`;
        console.warn("tile", key, "failed:", e.message);
    } finally {
        inflight.delete(key);
    }
}

let ringBusy = false;
let ringFails = 0, ringRetryAt = 0;
async function pumpRing() {
    if (ringBusy || performance.now() < ringRetryAt) return;
    ringBusy = true;
    try {
        const d = await getJSON(
            `${SERVER}/ring?x=${player.pos[0]}&z=${player.pos[2]}&r=${RADIUS}`,
            DEADLINE_MS.ring, "ring");
        ringFails = 0;
        queued = d.tiles.filter((t) => t.state !== "ready").length;
        // nearest first: /ring is already sorted by distance, and the tile underfoot
        // is the one the player is about to need
        for (const t of d.tiles) {
            if (t.state === "ready") fetchTile(t.tile[0], t.tile[1], t.origin_m);
        }
    } catch (e) {
        ringFails++;
        ringRetryAt = performance.now() + backoff(ringFails, 500, 8000);
        netNote = `ring failed (${e.message})`;
        console.warn("ring failed:", e.message);
    } finally {
        ringBusy = false;
    }
}

// ---------------------------------------------------------------- boot
let player = null;

async function boot() {
    SPEC = await getJSON(`${SERVER}/spec`, DEADLINE_MS.spec, "spec");
    console.log("spec:", SPEC);
    occ.cellM = SPEC.cell_m;

    // spawn on the centre cell of tile (0,0), which is where the server's own warm-up
    // ring is built, so the first frame already has geometry
    const mid = Math.floor(SPEC.span / 2);
    const [sx, sz] = cellCentre(mid, mid);
    player = new Player({ pos: [sx, 0, sz], yaw: 0, eyeHeight: SPEC.eye_h_m });

    await ensureMaze(mid, mid);
    await pumpRing();

    const { keys } = bindInput(canvas, player, {
        onLockChange: (locked) => { hud.hint.style.display = locked ? "none" : "block"; },
    });

    gl.clearColor(0.02, 0.02, 0.03, 1);
    let last = performance.now(), fps = 0, acc = 0, frames = 0, ringAcc = 0;
    let fogTarget = 0.010;

    function frame(now) {
        const dt = Math.min((now - last) / 1000, 0.1);
        last = now;
        acc += dt; frames++;
        if (acc > 0.5) { fps = frames / acc; acc = 0; frames = 0; }

        player.update(dt, keys, blocked);

        // Speed governor: if the tile the player is walking INTO is not resident yet,
        // cap speed at 25% until it lands. Sprinting into a generating tile is how
        // "the world flickers as it loads" happens; feet may not outrun the server.
        {
            const f = player.forward();
            const axp = player.pos[0] + f[0] * 2.0, azp = player.pos[2] + f[2] * 2.0;
            const [ac, ar] = cellAt(axp, azp);
            const aheadKey = `${Math.floor(ac / SPEC.span)},${Math.floor(ar / SPEC.span)}`;
            const readyAhead = haveTile.has(aheadKey);
            player.speedScale = readyAhead ? 1 : 0.25;
            // Fog is the loading curtain: thick while the tile ahead generates,
            // releasing slowly once it lands. Hides the pop-in, sells the tomb.
            fogTarget = readyAhead ? 0.010 : 0.11;
        }
        field.fogDensity += (fogTarget - field.fogDensity) * (1 - Math.exp(-1.6 * dt));

        ringAcc += dt;
        if (ringAcc > 0.35) {
            ringAcc = 0;
            const [c, r] = cellAt(player.pos[0], player.pos[2]);
            ensureMaze(c, r);
            pumpRing();
            for (const [key, slot] of field.slotOf) {
                const [tc, tr] = key.split(",").map(Number);
                const ox = (2 * (tc * SPEC.span) + SPEC.span - 1 + 1.5) * SPEC.cell_m;
                const oz = (2 * (tr * SPEC.span) + SPEC.span - 1 + 1.5) * SPEC.cell_m;
                field.slotCost[slot] = tileCost([ox, oz]);
            }
            // a tile evicted from the field must be re-fetchable, or walking back
            // leaves a hole that never fills
            for (const key of [...haveTile]) if (!field.slotOf.has(key)) haveTile.delete(key);
        }

        const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
        const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        const view = m4.viewFromPlayer(player);
        const proj = m4.perspective(1.32, canvas.width / canvas.height, 0.05, 200);
        field.requestSort(view);

        const drawn = field.draw(view, proj, [canvas.width, canvas.height], gain, dilate);
        drawDust(view, proj, [player.pos[0], 1.6, player.pos[2]], now / 1000);

        const [c, r] = cellAt(player.pos[0], player.pos[2]);
        const tc = Math.floor(c / SPEC.span), tr = Math.floor(r / SPEC.span);
        hud.stats.textContent =
            `${fps.toFixed(0)} fps | cell ${c},${r} | tile ${tc},${tr} | ` +
            `${field.count.toLocaleString()} splats (${drawn.toLocaleString()} drawn) | ` +
            `${field.slotOf.size}/${field.slotCount} tiles resident, ${served} fetched` +
            `${field.evictions ? `, ${field.evictions} evicted` : ""}`;
        hud.queue.textContent = netNote
            ? netNote
            : queued ? `generating ${queued} tile${queued > 1 ? "s" : ""} ahead of you` : "";

        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

window.addEventListener("error", (e) => {
    fatal("Runtime error", `${e.message}\n\n${(e.error && e.error.stack) || ""}`);
});

boot().catch((e) => {
    fatal(`Cannot reach the endless server at ${SERVER}`,
          `${e.message}\n\nStart it with:\n  conda activate hyworld2\n` +
          `  python ~/HY-World-2.0/experiments/labyrinth/endless/stream_server.py \\\n` +
          `      --port 8092 --max-splats 400000 --warm 1`);
});
