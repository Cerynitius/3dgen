// Real 3D collision against the rendered triangles.
//
// The previous collider re-derived boxes from the maze's wall flags, which meant
// collision and geometry were two independent descriptions of the same level. Any
// disagreement between them is felt as an "empty air wall": you stop against
// nothing, or walk through something you can see. This collider reads the very
// triangle soup that is drawn, so the two cannot drift apart — if you can see it,
// it stops you, and if you cannot, it does not.
//
// The body is a vertical stack of spheres (feet/waist/head) tested against every
// triangle in the neighbouring grid buckets. That gives genuine 3D clearance:
// low openings block the head, the floor and ceiling are ignored automatically
// because no body sphere reaches them, and nothing depends on walls being
// axis-aligned.

import { STRIDE } from "./geometry.js";

/** Squared distance from point p to triangle (a,b,c), and the closest point. */
export function closestPointOnTriangle(p, a, b, c) {
    const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
    const acx = c[0] - a[0], acy = c[1] - a[1], acz = c[2] - a[2];
    const apx = p[0] - a[0], apy = p[1] - a[1], apz = p[2] - a[2];
    const d1 = abx * apx + aby * apy + abz * apz;
    const d2 = acx * apx + acy * apy + acz * apz;
    if (d1 <= 0 && d2 <= 0) return a;

    const bpx = p[0] - b[0], bpy = p[1] - b[1], bpz = p[2] - b[2];
    const d3 = abx * bpx + aby * bpy + abz * bpz;
    const d4 = acx * bpx + acy * bpy + acz * bpz;
    if (d3 >= 0 && d4 <= d3) return b;

    const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        const v = d1 / (d1 - d3);
        return [a[0] + abx * v, a[1] + aby * v, a[2] + abz * v];
    }

    const cpx = p[0] - c[0], cpy = p[1] - c[1], cpz = p[2] - c[2];
    const d5 = abx * cpx + aby * cpy + abz * cpz;
    const d6 = acx * cpx + acy * cpy + acz * cpz;
    if (d6 >= 0 && d5 <= d6) return c;

    const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        const w = d2 / (d2 - d6);
        return [a[0] + acx * w, a[1] + acy * w, a[2] + acz * w];
    }

    const va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [b[0] + (c[0] - b[0]) * w, b[1] + (c[1] - b[1]) * w, b[2] + (c[2] - b[2]) * w];
    }

    const denom = 1 / (va + vb + vc);
    const v = vb * denom, w = vc * denom;
    return [a[0] + abx * v + acx * w, a[1] + aby * v + acy * w, a[2] + abz * v + acz * w];
}

const dist2 = (p, q) => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;

export class MeshCollider {
    /**
     * @param mesh interleaved Float32Array in geometry.js format
     * @param opts.bucket  grid bucket size in metres
     * @param opts.heights body sphere centre heights above the floor
     */
    constructor(mesh, opts = {}) {
        this.bucket = opts.bucket ?? 3.0;
        this.heights = opts.heights ?? [0.45, 1.05, 1.62];
        this.tris = [];
        for (let i = 0; i + 3 * STRIDE <= mesh.length; i += 3 * STRIDE) {
            const v = (k) => [mesh[i + k * STRIDE], mesh[i + k * STRIDE + 1], mesh[i + k * STRIDE + 2]];
            this.tris.push([v(0), v(1), v(2)]);
        }
        // uniform grid over XZ; a triangle lands in every bucket its AABB touches
        this.grid = new Map();
        this.tris.forEach((t, ti) => {
            const x0 = Math.min(t[0][0], t[1][0], t[2][0]), x1 = Math.max(t[0][0], t[1][0], t[2][0]);
            const z0 = Math.min(t[0][2], t[1][2], t[2][2]), z1 = Math.max(t[0][2], t[1][2], t[2][2]);
            for (let cz = Math.floor(z0 / this.bucket); cz <= Math.floor(z1 / this.bucket); cz++) {
                for (let cx = Math.floor(x0 / this.bucket); cx <= Math.floor(x1 / this.bucket); cx++) {
                    const key = cx + "," + cz;
                    let list = this.grid.get(key);
                    if (!list) this.grid.set(key, (list = []));
                    list.push(ti);
                }
            }
        });
    }

    /** Triangle indices near a world XZ position. */
    candidates(x, z, pad = 1) {
        const cx = Math.floor(x / this.bucket), cz = Math.floor(z / this.bucket);
        const out = [];
        for (let dz = -pad; dz <= pad; dz++) {
            for (let dx = -pad; dx <= pad; dx++) {
                const list = this.grid.get((cx + dx) + "," + (cz + dz));
                if (list) out.push(...list);
            }
        }
        return out;
    }

    /** Distance from a point to the nearest triangle (Infinity if none nearby). */
    nearestDistance(p) {
        let best = Infinity;
        for (const ti of this.candidates(p[0], p[2])) {
            const t = this.tris[ti];
            const q = closestPointOnTriangle(p, t[0], t[1], t[2]);
            const d = dist2(p, q);
            if (d < best) best = d;
        }
        return Math.sqrt(best);
    }

    /**
     * Does a body of radius r standing at (x, z) intersect the level?
     * Returns the deepest penetration found, or 0 when clear.
     */
    penetration(x, z, r) {
        let deepest = 0;
        const cands = this.candidates(x, z);
        for (const h of this.heights) {
            const p = [x, h, z];
            for (const ti of cands) {
                const t = this.tris[ti];
                const q = closestPointOnTriangle(p, t[0], t[1], t[2]);
                const d2v = dist2(p, q);
                if (d2v < r * r) {
                    const pen = r - Math.sqrt(d2v);
                    if (pen > deepest) deepest = pen;
                }
            }
        }
        return deepest;
    }

    blocked(x, z, r) { return this.penetration(x, z, r) > 0; }
}

/** Drop-in replacement for makeCollider(maze), backed by the rendered mesh. */
export function makeMeshCollider(mesh, opts = {}) {
    const mc = new MeshCollider(mesh, opts);
    const fn = (x, z, r) => mc.blocked(x, z, r);
    fn.collider = mc;
    return fn;
}
