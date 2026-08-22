// Does the client's collision agree with the server's geometry?
//
// This is the one disagreement that is invisible until it is embarrassing: the walls you
// see are reconstructed from the server's occupancy, and the walls you bump into are
// rebuilt here from wall bits. If the two rasterisations differ by one index the player
// walks through a wall, or into an empty corridor, and the splats look "wrong" when the
// reconstruction was fine all along.
//
//   node endless_selftest.mjs [http://127.0.0.1:8092]

import { Occupancy } from "./occupancy.js";

const SERVER = process.argv[2] || "http://127.0.0.1:8092";
let fails = 0;
const check = (name, ok, detail = "") => {
    console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
    if (!ok) fails++;
};

const spec = await (await fetch(`${SERVER}/spec`)).json();
const C0 = -12, R0 = -12, N = 32;
const win = await (await fetch(
    `${SERVER}/maze?c0=${C0}&r0=${R0}&cols=${N}&rows=${N}`)).json();

const occ = new Occupancy(spec.cell_m);
occ.ingest(win);

// the server rasterises the same box; ask it for the truth
const truth = await (await fetch(
    `${SERVER}/occupancy?c0=${C0}&r0=${R0}&cols=${N}&rows=${N}`)).json();

// The server's rasterisation is windowed, and it forces the outer ring solid on
// purpose -- that skirt stops a ray escaping the array and coming back as a black hole.
// It is an artefact of rendering a window, not geometry, and the client must NOT
// reproduce it: a skirt in the collider is an invisible wall across the corridor you
// were about to walk down. So the interior is compared exactly, and the ring is only
// required to be where the differences live.
const mine = occ.grid(C0, R0, N, N);
const H = truth.grid.length, W = truth.grid[0].length;
let inner = 0, ring = 0, firstInner = null;
for (let j = 0; j < H; j++)
    for (let i = 0; i < W; i++) {
        if (truth.grid[j][i] === mine[j][i]) continue;
        const onRing = j === 0 || i === 0 || j === H - 1 || i === W - 1;
        if (onRing) ring++;
        else {
            inner++;
            if (!firstInner) firstInner = `[${j}][${i}] server=${truth.grid[j][i]} client=${mine[j][i]}`;
        }
    }
check("client occupancy matches the server inside the window", inner === 0,
      `${inner} interior cells differ of ${(H - 2) * (W - 2)}` +
      `${firstInner ? ", first " + firstInner : ""}`);
// Note what this does NOT require: that the ring differs at all. It used to assert
// `ring > 0`, which fails whenever the client and server happen to agree everywhere --
// the best possible outcome scored as a FAIL. What matters is only that no difference
// escapes the ring, and `inner === 0` above already says that. This line reports the
// ring count so a sudden change in it is visible, and asserts the one thing that would
// actually be wrong: a difference somewhere that is neither interior nor on the ring.
check("no difference falls outside the server's window skirt", inner === 0,
      `${ring} of ${2 * (H + W) - 4} boundary cells differ, which the client ` +
      `deliberately does not reproduce`);

// every open cell centre must be walkable, or the player spawns inside a wall
let stuck = 0;
for (let r = R0 + 2; r < R0 + N - 2; r++)
    for (let c = C0 + 2; c < C0 + N - 2; c++) {
        const x = (2 * c + 1.5) * spec.cell_m, z = (2 * r + 1.5) * spec.cell_m;
        if (occ.blocked(x, z, 0.34)) stuck++;
    }
check("no open cell centre is blocked for a 0.34 m body", stuck === 0,
      `${stuck} blocked centres`);

// and the middle of a solid pillar must always block
let leaks = 0, pillars = 0;
for (let r = R0 + 2; r < R0 + N - 2; r++)
    for (let c = C0 + 2; c < C0 + N - 2; c++) {
        const x = (2 * c + 2.5) * spec.cell_m, z = (2 * r + 2.5) * spec.cell_m;  // even/even
        pillars++;
        if (!occ.blocked(x, z, 0.34)) leaks++;
    }
check("every pillar centre blocks", leaks === 0, `${leaks} of ${pillars} leaked`);

// A doorway the server says is open must be passable.
//
// The east bit of (c,r) and the west bit of (c+1,r) name the same slab, and solidIdx
// reads only the west one -- so testing the east bit alone silently assumes they always
// agree, which is exactly the asymmetry that would break the collider. Assert the
// agreement instead of assuming it, then test passability from the bit solidIdx uses.
let sealed = 0, doors = 0, asym = 0, firstAsym = null;
for (let r = R0 + 2; r < R0 + N - 2; r++)
    for (let c = C0 + 2; c < C0 + N - 2; c++) {
        const here = occ.walls.get(c + "," + r);
        const east = occ.walls.get((c + 1) + "," + r);
        if (!here || !east) continue;
        if (here[1] !== east[3]) {                      // E of (c,r) vs W of (c+1,r)
            asym++;
            if (!firstAsym) firstAsym = `cell ${c},${r} says E=${here[1]}, ` +
                                        `cell ${c + 1},${r} says W=${east[3]}`;
            continue;
        }
        if (here[1]) continue;                          // wall present, nothing to pass
        doors++;
        const x = (2 * c + 2.5) * spec.cell_m, z = (2 * r + 1.5) * spec.cell_m;
        if (occ.solidIdx(Math.floor(x / spec.cell_m), Math.floor(z / spec.cell_m))) sealed++;
    }
check("the shared wall between neighbours is one bit, not two", asym === 0,
      `${asym} disagreements${firstAsym ? ", first " + firstAsym : ""}`);
check("open east doorways are passable", sealed === 0, `${sealed} of ${doors} sealed`);

// The map is not allowed to grow forever: an endless world means an endless walk.
const before = occ.walls.size;
const droppedFar = occ.prune(C0 + N / 2, R0 + N / 2, 4);
check("prune drops distant cells", droppedFar > 0 && occ.walls.size === before - droppedFar,
      `${before} -> ${occ.walls.size} cells`);
const kept = occ.walls.size;
check("prune keeps everything inside its horizon", occ.prune(C0 + N / 2, R0 + N / 2, 4) === 0
      && occ.walls.size === kept, `${occ.walls.size} cells after a second pass`);

console.log();
if (fails) { console.log(`FAILED: ${fails}`); process.exit(1); }
console.log("ALL PASS");
