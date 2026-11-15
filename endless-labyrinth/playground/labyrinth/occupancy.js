// Client-side occupancy, rebuilt from the wall bits /maze serves.
//
// Kept free of browser globals on purpose: this is the one piece of the client whose
// disagreement with the server is invisible until you walk through a wall, so it has to
// be runnable under node and diffed against infinite_maze.occupancy() directly. See
// endless_selftest.mjs.
//
// The grid is the same odd/even rasterisation the renderer walks server-side: an open
// cell (c,r) sits at index (2c+1, 2r+1), the gaps between are doorways when the wall is
// absent, and even/even intersections are always pillars.

export const DIRS = { N: 0, E: 1, S: 2, W: 3 };

export class Occupancy {
    constructor(cellM = 2.0) {
        this.cellM = cellM;
        this.walls = new Map();          // "c,r" -> [N,E,S,W] booleans
        this.box = null;
    }

    /** Absorb one /maze window. */
    ingest(d) {
        for (let j = 0; j < d.rows; j++)
            for (let i = 0; i < d.cols; i++)
                this.walls.set((d.c0 + i) + "," + (d.r0 + j), d.walls[j * d.cols + i]);
        this.box = { c0: d.c0, r0: d.r0, c1: d.c0 + d.cols, r1: d.r0 + d.rows };
    }

    /**
     * Forget wall bits further than `half` cells from (c, r).
     *
     * `walls` is not a cache -- nothing ever removed from it. Every /maze window is
     * 56x56, one arrives roughly every 18 cells walked, and an endless world means an
     * endless walk, so the map grows without bound for as long as the tab is open. That
     * is slow enough to look like a leak rather than a bug and fast enough to matter in
     * a long session.
     *
     * `half` must stay comfortably larger than the window: an entry that is dropped
     * while still inside the loaded box reads back as "not loaded", solidIdx treats
     * unknown as solid, and the player walks into an invisible wall in a corridor that
     * is plainly open. Callers pass three window half-widths, which leaves the box
     * (at most 46 cells from the player) well clear of the 84-cell horizon.
     */
    prune(c, r, half) {
        const keep = (2 * half + 1) ** 2;
        if (this.walls.size <= keep) return 0;
        let dropped = 0;
        for (const key of this.walls.keys()) {
            const i = key.indexOf(",");
            if (Math.abs(+key.slice(0, i) - c) <= half
                && Math.abs(+key.slice(i + 1) - r) <= half) continue;
            this.walls.delete(key);
            dropped++;
        }
        return dropped;
    }

    wallAt(c, r, d) {
        const w = this.walls.get(c + "," + r);
        return w ? w[d] : null;          // null = not loaded here
    }

    /** True where the occupancy grid is solid. Unknown counts as solid. */
    solidIdx(ix, iz) {
        const xo = ((ix % 2) + 2) % 2, zo = ((iz % 2) + 2) % 2;
        if (xo === 1 && zo === 1) {
            // the cell itself: open wherever its walls are known at all
            return this.walls.has(((ix - 1) / 2) + "," + ((iz - 1) / 2)) ? false : true;
        }
        if (xo === 1 && zo === 0) {
            // doorway north of cell (c, iz/2)
            const w = this.wallAt((ix - 1) / 2, iz / 2, DIRS.N);
            return w === null ? true : w;
        }
        if (xo === 0 && zo === 1) {
            // doorway west of cell (ix/2, r)
            const w = this.wallAt(ix / 2, (iz - 1) / 2, DIRS.W);
            return w === null ? true : w;
        }
        return true;                     // pillar
    }

    /** Circle-vs-grid test for a body of radius r at (x,z) metres. */
    blocked(x, z, r) {
        const m = this.cellM, k = r * 0.7071;
        const probes = [[0, 0], [r, 0], [-r, 0], [0, r], [0, -r],
                        [k, k], [-k, k], [k, -k], [-k, -k]];
        for (const [ox, oz] of probes)
            if (this.solidIdx(Math.floor((x + ox) / m), Math.floor((z + oz) / m))) return true;
        return false;
    }

    /** Rasterise a cell box the way the server does, for direct comparison. */
    grid(c0, r0, cols, rows) {
        const out = [];
        for (let jz = 0; jz < 2 * rows + 1; jz++) {
            const row = [];
            for (let jx = 0; jx < 2 * cols + 1; jx++)
                row.push(this.solidIdx(2 * c0 + jx, 2 * r0 + jz) ? 1 : 0);
            out.push(row);
        }
        return out;
    }
}
