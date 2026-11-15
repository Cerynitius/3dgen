// Live world: walk the labyrinth while WorldMirror generates it around you.
//
// The layout is known locally (same seeded cell generator the server uses), so
// collision and the map are exact from the first frame. The *appearance* is streamed:
// as you approach a cell that has not been reconstructed, the chunk server renders
// views of it, runs one WorldMirror forward pass, and returns Gaussians, which are
// appended to the splat field. Measured at ~0.6s per 6-view chunk against a walking
// pace of 0.87s per cell, so generation stays ahead of the player.

import { Labyrinth } from "./maze.js";
import { Player, bindInput, BINDINGS, makeCollider } from "./controls.js";
import { losClear } from "./geometry.js";
import { SplatField } from "./splats.js";
import * as m4 from "./mat4.js";
import { withDeadline, Defer, backoff } from "./netpolicy.js";

const params = new URLSearchParams(location.search);

// A query string is user input, and an unvalidated one fails silently and expensively:
// ?gain=abc reaches the shader as a NaN uniform, which multiplies every alpha to NaN and
// paints a black screen with no error anywhere; ?max=abc sends max=NaN to the server.
// Both read as "the world stopped working". Unparseable and out-of-range both fall back.
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
const seed = numParam("seed", 1000, 0, 2 ** 31, parseInt);
const cols = numParam("cols", 14, 2, 512, parseInt);
const rows = numParam("rows", 14, 2, 512, parseInt);
// 8 = a stereo pair on each of the four sides, so no wall of the cell is unobserved
// 12 views at 30 deg steps = 57% overlap between consecutive frames, the standard
// photogrammetry target; below ~40% the reconstruction stops fusing into one scene
const views = numParam("views", 18, 2, 64, parseInt);
// One reconstruction per SPANxSPAN region. Per-cell chunks each self-normalise, so
// neighbours disagree and their faces cross; a region is one consistent pass.
const SPAN = numParam("span", 3, 1, 8, parseInt);
// Keeping only a fraction of the splats punches holes in the coverage, which reads
// as a dark, see-through world. Full chunks are ~66MB; tune with ?max= if streaming
// over wifi rather than localhost.
// 333,312 = one slot at 8M capacity. Denser chunks matter more than more resident
// cells: eviction is by distance and the visible set is rarely over ~12 cells.
const maxSplats = numParam("max", 333312, 1000, 8_000_000, parseInt);
// 952 is the official CLI default; I had been running 518, barely half the linear
// resolution, which caps how much wall detail can exist at all
const size = numParam("size", 952, 128, 2048, parseInt);
// gain/dilate were compensations for a sparse, low-res field. At 952 with the
// oversized splats filtered the field is dense enough to composite honestly, and
// leaving them inflated is what makes the scene look cloudy and fogged.
const gain = numParam("gain", 1.4, 0.1, 8.0);
// Past ~1.5m every splat is sub-pixel, so this floor is what actually draws distant
// geometry. 0.6 (~2.3px radius) left gaps between neighbours -> visible dots; 1.2
// (~3.3px) overlaps them into surfaces at the cost of slight softness up close.
const dilate = numParam("dilate", 0.35, 0.0, 4.0);   // ~standard 3DGS low-pass
const RADIUS_UNUSED = numParam("radius", 3, 1, 16, parseInt);   // cells kept generated around you;
// 2 only reached ~6m, which capped how far you could see regardless of lighting
const SERVER = params.get("server") ||
    `${location.protocol}//${location.hostname}:8091`;

// MUST match labyrinth_world.py's metric scale, or chunks are placed on a grid with a
// different pitch than the one they were generated in and drift 1 m per cell away from
// spawn. Python: open cell = 1 grid unit of CELL_M=2 m, walls a full grid unit thick,
// so pitch = 4 m and corridors are 2 m wide.
const PY_CELL = 4.0, PY_WALL = 2.0;
const maze = new Labyrinth({ cols, rows, seed, cell: PY_CELL, wallT: PY_WALL });
// Collide against the wall SLABS, as solid volumes.
//
// This used to be makeMeshCollider(buildLevelMesh(maze)), and buildLevelMesh emits only
// the single inward-facing plane of each wall -- it is a render shell, not a solid. Two
// such planes face each other across a 2 m cavity, and nothing caps the ends of a slab
// at a T-junction or a dead end, so a 0.34 m body walks in through the open end and is
// then inside the wall. wallsNear returns the slab itself as an AABB, which cannot be
// entered from any direction; probed against an unwindowed all-walls collider it agrees
// on 1,609,944 of 1,609,944 samples across six seeds, so the span=1 window costs nothing.
const collide = makeCollider(maze);
const spawn = maze.centre(maze.spawn.c, maze.spawn.r);
const player = new Player({ pos: spawn, yaw: 0 });

const canvas = document.getElementById("view");
const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
const hud = {
    stats: document.getElementById("stats"),
    queue: document.getElementById("queue"),
    fatal: document.getElementById("fatal"),
};
if (!gl) {
    hud.fatal.style.display = "block";
    hud.fatal.textContent = "WebGL2 is required.";
    throw new Error("no webgl2");
}
if (!gl.getExtension("EXT_color_buffer_float")) {
    console.warn("EXT_color_buffer_float missing; float textures may be limited");
}

// A slot must hold the LARGEST chunk the server can emit, not the budget that was asked
// for: voxel pruning accepts anything within 1.1x of its target. 24 slots of a 8M
// capacity is 333,312 each -- exactly the default max, and 10% under the worst case --
// so chunks were silently cut short, and cutting a spatially uniform pruned list short
// removes coverage in patches rather than uniformly: holes in otherwise solid walls.
// Deriving the slot count from the budget keeps that true when ?max= changes it.
const SLOTS = Math.max(4, Math.floor(8_000_000 / Math.ceil(maxSplats * 1.1)));
const field = new SplatField(gl, 8_000_000, SLOTS);
gl.clearColor(0.02, 0.02, 0.03, 1);

// ---------------------------------------------------------------- streaming
const requested = new Set();
const done = new Set();
const defer = new Defer();               // see netpolicy.js
let inflight = 0, failed = 0, lastMs = 0, totalMs = 0, chunks = 0;
const MAX_INFLIGHT = 2;
const CHUNK_DEADLINE_MS = 120000;

async function fetchChunk(c, r) {
    const key = `${c},${r}`;
    if (requested.has(key) || defer.has(key, chunkCost(c, r))) return;
    requested.add(key);
    inflight++;
    const t0 = performance.now();
    try {
        const url = `${SERVER}/chunk?seed=${seed}&cols=${cols}&rows=${rows}` +
                    `&c=${c}&r=${r}&views=${views}&max=${maxSplats}&size=${size}&span=${SPAN}`;
        const got = await withDeadline(CHUNK_DEADLINE_MS, `chunk ${key}`, async (signal) => {
            const res = await fetch(url, { signal });
            if (!res.ok) throw new Error("HTTP " + res.status);
            const meta = JSON.parse(res.headers.get("X-Chunk-Meta") || "{}");
            return { meta, buf: await res.arrayBuffer() };
        });
        // The slot is keyed on the REQUESTED cell but placed by the response's own
        // meta.cell, and evicted by a cost computed from the request again. chunk_server
        // echoes the request verbatim so the three agree today -- but if it ever
        // normalised or shifted a region, splats would be positioned as one cell and
        // registered as another, and the only symptom would be chunks landing in the
        // wrong place and being re-downloaded forever. Assert the contract instead of
        // depending on it quietly.
        if (Array.isArray(got.meta.cell) && (got.meta.cell[0] !== c || got.meta.cell[1] !== r))
            throw new Error(`server answered for cell ${got.meta.cell} but ${key} was asked for`);
        const added = field.addChunk(key, new Float32Array(got.buf), chunkTransform(got.meta),
                                     chunkCost(c, r));
        const meta = got.meta, buf = got.buf;
        if (added === 0) {
            // The field refused it: every resident chunk is nearer. Dropping the key
            // straight out of `requested` put the same tens of megabytes back on the
            // wire 0.25 s later, and again, forever -- SCAN=8/SPAN=3 offers ~96
            // candidates into 21 slots, so this is the steady state, not an edge case.
            // Only walking closer can change the answer, so that is what ends the wait.
            requested.delete(key);
            defer.hold(key, () => 3000, chunkCost(c, r));
            return;
        }
        defer.clear(key);
        done.add(key);
        chunks++;
        lastMs = performance.now() - t0;
        totalMs += lastMs;
        console.log(`chunk ${key}: ${added.toLocaleString()} splats in ${lastMs.toFixed(0)}ms`,
                    meta.timing);
    } catch (err) {
        failed++;
        requested.delete(key);                 // let it be retried later
        // ...but not immediately, and not at 0.25 s forever. A server that is failing
        // was being hit by every candidate cell several times a second.
        defer.hold(key, (n) => backoff(n, 500, 8000));
        console.warn("chunk", key, "failed:", err.message);
    } finally {
        inflight--;
    }
}

/**
 * Which cell to generate next.
 *
 * A radius sweep is the wrong shape for a labyrinth: most cells within 3 of you are
 * behind a wall and will never be seen, while the corridor you are looking down runs
 * 8 cells out and is exactly what reads as "limited view range". This scores
 * candidates instead of sweeping them:
 *
 *   visible   line of sight through open walls  -- the only cells that can be seen
 *   facing    inside the view cone, weighted by how central
 *   lead      near where you will BE in ~1.5s, not where you are
 *   near      closer is cheaper to be wrong about
 *
 * Invisible cells are still generated, last, so turning around is not a stall.
 */
const SCAN = 8;                      // how far down a corridor to look
const LEAD_TIME = 1.5;               // seconds of movement to extrapolate

function cellScore(c, r, eye, fwd, lead) {
    const centre = maze.centre(c, r);
    const dx = centre[0] - eye[0], dz = centre[2] - eye[2];
    const dist = Math.hypot(dx, dz) || 1e-3;
    const facing = (dx / dist) * fwd[0] + (dz / dist) * fwd[2];      // -1..1
    const visible = losClear(maze, eye[0], eye[2], centre[0], centre[2]);
    const leadDist = Math.hypot(centre[0] - lead[0], centre[2] - lead[2]);
    let score = dist * 0.6 + leadDist * 0.8;                          // lower is better
    if (visible) score -= 14;                                          // dominates
    if (facing > 0.25) score -= 6 * facing;
    if (!visible && facing < 0) score += 10;                           // behind you, later
    return score;
}

function pump() {
    if (inflight >= MAX_INFLIGHT) return;
    const eye = player.eye();
    const fwd = player.forward();
    const lead = [eye[0] + player.vel[0] * LEAD_TIME, 0, eye[2] + player.vel[1] * LEAD_TIME];
    const here = maze.cellAt(player.pos[0], player.pos[2]);

    const cands = [];
    for (let dr = -SCAN; dr <= SCAN; dr++) {
        for (let dc = -SCAN; dc <= SCAN; dc++) {
            const cc = here.c + dc, rr = here.r + dr;
            if (!maze.inside(cc, rr) || !maze.carved[maze.idx(cc, rr)]) continue;
            const c = Math.floor(cc / SPAN) * SPAN, r = Math.floor(rr / SPAN) * SPAN;
            const key = `${c},${r}`;
            if (requested.has(key) || cands.some((x) => x.key === key)) continue;
            cands.push({ c, r, key, s: cellScore(cc, rr, eye, fwd, lead) });
        }
    }
    cands.sort((a, b) => a.s - b.s);
    while (inflight < MAX_INFLIGHT && cands.length) {
        const w = cands.shift();
        fetchChunk(w.c, w.r);
    }
}

/** Eviction cost: distance, so cells you have walked away from are reclaimed first. */
function chunkCost(c, r) {
    const centre = maze.centre(c, r);
    return Math.hypot(centre[0] - player.pos[0], centre[2] - player.pos[2]);
}

const { keys } = bindInput(canvas, player, {
    onPress: (code) => {
        if (code === "KeyR") location.search =
            `?seed=${Math.floor(Math.random() * 1e6)}&cols=${cols}&rows=${rows}`;
    },
    onLockChange: (locked) => {
        document.getElementById("hint").style.display = locked ? "none" : "block";
    },
});

const BASE_SPEED = player.walkSpeed;
let last = performance.now(), fps = 0, acc = 0, frames = 0, pumpAcc = 0;

function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    acc += dt; frames++;
    if (acc > 0.5) { fps = frames / acc; acc = 0; frames = 0; }

    // One steady pace the world can keep up with: the lower of full speed and what
    // generation sustains (cell pitch / chunk seconds), floored so you never stop.
    const MIN_SPEED = 0.7;
    const chunkSec = chunks ? (totalMs / chunks) / 1000 : 1.0;
    const vSustain = Math.max(MIN_SPEED, maze.cell / Math.max(chunkSec, 1e-3));
    const target = Math.min(BASE_SPEED, vSustain);
    player.walkSpeed += (target - player.walkSpeed) * Math.min(1, 3 * dt);

    // Collide against the maze, always.
    //
    // This used to switch to field.blocked() -- the occupancy of the splats themselves --
    // the moment a chunk arrived, on the reasoning that you should stop at what you can
    // see. The reconstruction has holes, so that makes every hole in a wall a hole you
    // can walk through, and the rule changed underfoot at the instant a chunk landed:
    // stand in the wall cavity when the chunk arrives and the wall's own voxels close
    // around you. endless.js settled this the other way for exactly these reasons, and
    // this is the same decision. Fixing the reconstruction is the answer to a visible
    // hole; making the body fall through it is not.
    player.update(dt, keys, collide);

    pumpAcc += dt;
    if (pumpAcc > 0.25) {
        pumpAcc = 0;
        for (const [key, slot] of field.slotOf) {
            const [cc, rr] = key.split(",").map(Number);
            field.slotCost[slot] = chunkCost(cc, rr);
        }
        for (const key of [...requested]) if (!field.slotOf.has(key) && done.has(key)) {
            requested.delete(key); done.delete(key);
        }
        pump();
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

    const cell = maze.cellAt(player.pos[0], player.pos[2]);
    hud.stats.textContent =
        `${fps.toFixed(0)} fps | cell ${cell.c},${cell.r} | ` +
        `${field.count.toLocaleString()} splats (${drawn.toLocaleString()} drawn) | ` +
        `cells ${field.slotOf.size}/${field.slotCount} slots, ${done.size} built` +
        `${field.evictions ? `, ${field.evictions} evicted` : ""}`;
    const pending = requested.size - done.size;
    hud.queue.textContent = player.walkSpeed < BASE_SPEED * 0.98
        ? `${player.walkSpeed.toFixed(1)} m/s, matched to ${chunkSec.toFixed(1)}s/cell generation`
        : pending ? `generating ${pending} cell${pending > 1 ? "s" : ""} (walk on, they pop in)` : "";

    requestAnimationFrame(frame);
}

// Health check and startup are kept separate: a single .catch() around both reported
// every runtime crash in frame() as "cannot reach the chunk server", which sent the
// last debugging session after a server that was working fine.
fetch(`${SERVER}/health`)
    .then((r) => r.json())
    .then((h) => { console.log("chunk server:", h); })
    .catch((e) => {
        hud.fatal.style.display = "block";
        hud.fatal.textContent =
            `Cannot reach the chunk server at ${SERVER}\n\n${e.message}\n\n` +
            `Start it with:\n  conda activate hyworld2\n` +
            `  python ~/HY-World-2.0/experiments/labyrinth/chunk_server.py --warm`;
    });

// Runtime errors surface as themselves, with a stack, not as a network diagnosis.
window.addEventListener("error", (e) => {
    hud.fatal.style.display = "block";
    hud.fatal.textContent = `Runtime error\n\n${e.message}\n\n${(e.error && e.error.stack) || ""}`;
});

try {
    pump();
    requestAnimationFrame(frame);
} catch (e) {
    hud.fatal.style.display = "block";
    hud.fatal.textContent = `Startup failed\n\n${e.message}\n\n${e.stack || ""}`;
    throw e;
}
