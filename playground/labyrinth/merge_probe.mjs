// Probe: compare buildLevelMesh (per-cell quads) vs a run-merged variant.
// Verifies identical coverage (total floor/wall area) and counts the savings.
import { buildLevelMesh } from '/tmp/lab_new/geometry.js';
import { Labyrinth } from '/tmp/lab_new/maze.js';

const STRIDE = 9;

// --- merged implementation ---
function buildLevelMeshMerged(maze) {
    const v = [];
    const S = maze.cell, T = maze.wallT, H = maze.wallH;
    const FLOOR = 0, CEIL = 1, WALL = 2;
    const quad = (a, b, c, d, n, uv, surf) => {
        // same winding/layout as MeshBuilder.quad: two triangles a-b-c, a-c-d
        const push = (p, u2) => v.push(p[0], p[1], p[2], n[0], n[1], n[2], u2[0], u2[1], surf);
        push(a, uv[0]); push(b, uv[1]); push(c, uv[2]); push(a, uv[0]); push(c, uv[2]); push(d, uv[3]);
    };

    // floor + ceiling: merge vertical runs of carved cells per column
    for (let c = 0; c < maze.cols; c++) {
        let r = 0;
        while (r < maze.rows) {
            if (!maze.carved[maze.idx(c, r)]) { r++; continue; }
            let r1 = r;
            while (r1 + 1 < maze.rows && maze.carved[maze.idx(c, r1 + 1)]) r1++;
            const x0 = c * S, x1 = x0 + S, z0 = r * S, z1 = (r1 + 1) * S;
            quad([x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1], [0, 1, 0],
                 [[x0, z0], [x1, z0], [x1, z1], [x0, z1]], FLOOR);
            quad([x0, H, z1], [x1, H, z1], [x1, H, z0], [x0, H, z0], [0, -1, 0],
                 [[x0, z1], [x1, z1], [x1, z0], [x0, z0]], CEIL);
            r = r1 + 1;
        }
    }
    // north/south walls: merge horizontal runs per row
    for (let r = 0; r < maze.rows; r++) {
        for (const dir of [0, 2]) {
            let c = 0;
            while (c < maze.cols) {
                if (!maze.carved[maze.idx(c, r)] || !maze.hasWall(c, r, dir)) { c++; continue; }
                let c1 = c;
                while (c1 + 1 < maze.cols && maze.carved[maze.idx(c1 + 1, r)] && maze.hasWall(c1 + 1, r, dir)) c1++;
                const x0 = c * S, x1 = (c1 + 1) * S;
                const z0 = r * S, z1 = z0 + S;
                if (dir === 0) {
                    const z = z0 + T / 2;
                    quad([x0, 0, z], [x1, 0, z], [x1, H, z], [x0, H, z], [0, 0, 1],
                         [[x0, 0], [x1, 0], [x1, H], [x0, H]], WALL);
                } else {
                    const z = z1 - T / 2;
                    quad([x1, 0, z], [x0, 0, z], [x0, H, z], [x1, H, z], [0, 0, -1],
                         [[x1, 0], [x0, 0], [x0, H], [x1, H]], WALL);
                }
                c = c1 + 1;
            }
        }
    }
    // west/east walls: merge vertical runs per column
    for (let c = 0; c < maze.cols; c++) {
        for (const dir of [3, 1]) {
            let r = 0;
            while (r < maze.rows) {
                if (!maze.carved[maze.idx(c, r)] || !maze.hasWall(c, r, dir)) { r++; continue; }
                let r1 = r;
                while (r1 + 1 < maze.rows && maze.carved[maze.idx(c, r1 + 1)] && maze.hasWall(c, r1 + 1, dir)) r1++;
                const z0 = r * S, z1 = (r1 + 1) * S;
                const x0 = c * S, x1 = x0 + S;
                if (dir === 3) {
                    const x = x0 + T / 2;
                    quad([x, 0, z1], [x, 0, z0], [x, H, z0], [x, H, z1], [1, 0, 0],
                         [[z1, 0], [z0, 0], [z0, H], [z1, H]], WALL);
                } else {
                    const x = x1 - T / 2;
                    quad([x, 0, z0], [x, 0, z1], [x, H, z1], [x, H, z0], [-1, 0, 0],
                         [[z0, 0], [z1, 0], [z1, H], [z0, H]], WALL);
                }
                r = r1 + 1;
            }
        }
    }
    return new Float32Array(v);
}

// --- area accounting per surface type ---
function areas(buf) {
    const out = { 0: 0, 1: 0, 2: 0, quads: 0 };
    for (let i = 0; i < buf.length; i += STRIDE * 6) {
        const surf = buf[i + 8];
        // quad = verts a,b,c,d at strides 0,1,3 (a,b,d are the rect corners)
        const ax = buf[i], ay = buf[i+1], az = buf[i+2];
        const bx = buf[i+STRIDE], bz = buf[i+STRIDE+2];
        const dx = buf[i+STRIDE*3], dz = buf[i+STRIDE*3+2];
        const w = Math.hypot(bx - ax, bz - az), h2 = Math.hypot(dx - ax, dz - az);
        // wall quads: one edge is horizontal (w), other vertical H — use full 3D
        const by = buf[i+STRIDE+1], dy = buf[i+STRIDE*3+1];
        const w3 = Math.hypot(bx-ax, by-ay, bz-az), h3 = Math.hypot(dx-ax, dy-ay, dz-az);
        out[surf] += w3 * h3;
        out.quads++;
    }
    return out;
}

let totOld = 0, totNew = 0, fails = 0;
for (const seed of [1000, 7, 42, 1337, 2026]) {
    const maze = new Labyrinth({ seed, cols: 14, rows: 14 });
    const a = areas(buildLevelMesh(maze));
    const b = areas(buildLevelMeshMerged(maze));
    const ok = Math.abs(a[0]-b[0]) < 1e-3 && Math.abs(a[1]-b[1]) < 1e-3 && Math.abs(a[2]-b[2]) < 1e-3;
    if (!ok) fails++;
    totOld += a.quads; totNew += b.quads;
    console.log(`seed=${seed}  quads ${a.quads} -> ${b.quads}  (${(a.quads/b.quads).toFixed(1)}x fewer)  area match: ${ok}`);
}
console.log(fails === 0 ? `\nTOTAL quads ${totOld} -> ${totNew}, all areas identical` : `\n${fails} seeds MISMATCHED`);
