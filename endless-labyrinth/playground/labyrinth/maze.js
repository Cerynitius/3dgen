// Deterministic cell-based labyrinth generator.
//
// The unit of the world is a CELL, not an occupancy pixel. Every cell carries its
// own four wall flags, a kind, and its torch bracket anchors, so geometry, lighting
// and gameplay all read from one structure. Shared walls are always mutated through
// carve()/seal() so the two cells that meet at a wall can never disagree.
//
// Pipeline: rooms -> growing-tree maze over the leftover cells -> doors joining the
// rooms in -> braiding (dead-end removal) -> connectivity assert -> classification
// -> torch placement -> spawn/goal selection.

export const N = 0, E = 1, S = 2, W = 3;
export const DIRS = [N, E, S, W];
export const DC = [0, 1, 0, -1];   // + column when moving E
export const DR = [-1, 0, 1, 0];   // - row when moving N
export const OPP = [S, W, N, E];
export const DIR_NAME = ["N", "E", "S", "W"];

// mulberry32 — small, fast, and reproducible across Node and the browser.
export function makeRng(seed) {
    let a = seed >>> 0;
    return function () {
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const pick = (rnd, arr) => arr[Math.floor(rnd() * arr.length) % arr.length];

export class Labyrinth {
    constructor(opts = {}) {
        this.cols = opts.cols ?? 24;
        this.rows = opts.rows ?? 24;
        this.seed = opts.seed ?? 1337;
        this.cell = opts.cell ?? 3.0;          // metres across a cell
        this.wallH = opts.wallH ?? 3.2;        // floor -> ceiling
        this.wallT = opts.wallT ?? 0.28;       // wall slab thickness
        this.braid = opts.braid ?? 0.38;       // fraction of dead ends reopened
        this.roomTries = opts.roomTries ?? 14;
        this.torchSpacing = opts.torchSpacing ?? 3;  // min cells between brackets
        this.rnd = makeRng(this.seed);

        const n = this.cols * this.rows;
        // bit d set => wall present on side d
        this.walls = new Uint8Array(n).fill(0b1111);
        this.kind = new Array(n).fill("solid");   // solid|corridor|room|junction|dead-end|corner
        this.room = new Int16Array(n).fill(-1);
        this.carved = new Uint8Array(n);          // 1 once the cell is part of the maze
        this.rooms = [];
        this.torches = [];
        this.spares = [];
        this.generate();
    }

    idx(c, r) { return r * this.cols + c; }
    inside(c, r) { return c >= 0 && r >= 0 && c < this.cols && r < this.rows; }
    hasWall(c, r, d) { return (this.walls[this.idx(c, r)] & (1 << d)) !== 0; }

    /** Open the wall between (c,r) and its neighbour on side d, on both cells. */
    carve(c, r, d) {
        const nc = c + DC[d], nr = r + DR[d];
        if (!this.inside(nc, nr)) return false;
        this.walls[this.idx(c, r)] &= ~(1 << d);
        this.walls[this.idx(nc, nr)] &= ~(1 << OPP[d]);
        return true;
    }

    /** Close the wall between (c,r) and side d, on both cells. */
    seal(c, r, d) {
        this.walls[this.idx(c, r)] |= (1 << d);
        const nc = c + DC[d], nr = r + DR[d];
        if (this.inside(nc, nr)) this.walls[this.idx(nc, nr)] |= (1 << OPP[d]);
    }

    neighbours(c, r) {
        const out = [];
        for (const d of DIRS) {
            const nc = c + DC[d], nr = r + DR[d];
            if (this.inside(nc, nr)) out.push({ d, c: nc, r: nr, i: this.idx(nc, nr) });
        }
        return out;
    }

    /** Sides of (c,r) that are open. */
    exits(c, r) { return DIRS.filter((d) => !this.hasWall(c, r, d)); }

    // ---------------------------------------------------------------- rooms
    carveRooms() {
        for (let attempt = 0; attempt < this.roomTries; attempt++) {
            const w = 2 + Math.floor(this.rnd() * 3);   // 2..4
            const h = 2 + Math.floor(this.rnd() * 3);
            const c0 = 1 + Math.floor(this.rnd() * (this.cols - w - 2));
            const r0 = 1 + Math.floor(this.rnd() * (this.rows - h - 2));
            if (c0 < 1 || r0 < 1) continue;
            // reject if it touches an existing room (keep a 1-cell rock margin)
            let clash = false;
            for (let r = r0 - 1; r <= r0 + h && !clash; r++)
                for (let c = c0 - 1; c <= c0 + w; c++)
                    if (this.inside(c, r) && this.room[this.idx(c, r)] >= 0) { clash = true; break; }
            if (clash) continue;

            const id = this.rooms.length;
            for (let r = r0; r < r0 + h; r++) {
                for (let c = c0; c < c0 + w; c++) {
                    const i = this.idx(c, r);
                    this.room[i] = id;
                    this.carved[i] = 1;
                    this.kind[i] = "room";
                    if (c + 1 < c0 + w) this.carve(c, r, E);
                    if (r + 1 < r0 + h) this.carve(c, r, S);
                }
            }
            this.rooms.push({ id, c0, r0, w, h });
        }
    }

    // ---------------------------------------------------------------- maze
    /** Growing-tree over every cell the rooms did not take. */
    growMaze() {
        let start = null;
        for (let r = 0; r < this.rows && !start; r++)
            for (let c = 0; c < this.cols; c++)
                if (!this.carved[this.idx(c, r)]) { start = { c, r }; break; }
        while (start) {
            const stack = [start];
            this.carved[this.idx(start.c, start.r)] = 1;
            this.kind[this.idx(start.c, start.r)] = "corridor";
            while (stack.length) {
                const cur = stack[stack.length - 1];
                const open = this.neighbours(cur.c, cur.r).filter(
                    (nb) => !this.carved[nb.i] && this.room[nb.i] < 0);
                if (!open.length) { stack.pop(); continue; }
                // bias towards carrying straight on, which reads as corridors not mush
                const straight = stack.length > 1
                    ? open.find((nb) => nb.d === cur.from) : null;
                const nb = straight && this.rnd() < 0.62 ? straight : pick(this.rnd, open);
                this.carve(cur.c, cur.r, nb.d);
                this.carved[nb.i] = 1;
                this.kind[nb.i] = "corridor";
                stack.push({ c: nb.c, r: nb.r, from: nb.d });
            }
            // next untouched cell, if any
            start = null;
            for (let r = 0; r < this.rows && !start; r++)
                for (let c = 0; c < this.cols; c++)
                    if (!this.carved[this.idx(c, r)] && this.room[this.idx(c, r)] < 0) {
                        start = { c, r }; break;
                    }
        }
    }

    // ---------------------------------------------------------------- doors
    /** Punch 1-3 doors per room into whatever lies outside it. */
    connectRooms() {
        for (const room of this.rooms) {
            const cands = [];
            for (let r = room.r0; r < room.r0 + room.h; r++) {
                for (let c = room.c0; c < room.c0 + room.w; c++) {
                    for (const d of DIRS) {
                        const nc = c + DC[d], nr = r + DR[d];
                        if (!this.inside(nc, nr)) continue;
                        const ni = this.idx(nc, nr);
                        if (this.room[ni] === room.id) continue;   // interior wall
                        if (!this.carved[ni]) continue;            // solid rock
                        cands.push({ c, r, d });
                    }
                }
            }
            if (!cands.length) continue;
            const doors = 1 + Math.floor(this.rnd() * 3);
            const used = new Set();
            for (let k = 0; k < doors && cands.length; k++) {
                const pickd = pick(this.rnd, cands);
                const key = `${pickd.c},${pickd.r},${pickd.d}`;
                if (used.has(key)) continue;
                used.add(key);
                this.carve(pickd.c, pickd.r, pickd.d);
            }
        }
    }

    // ---------------------------------------------------------------- braid
    /** Reopen a fraction of dead ends so the maze has loops instead of one tree. */
    braidDeadEnds() {
        const ends = [];
        for (let r = 0; r < this.rows; r++)
            for (let c = 0; c < this.cols; c++) {
                const i = this.idx(c, r);
                if (this.carved[i] && this.exits(c, r).length === 1) ends.push({ c, r });
            }
        for (const e of ends) {
            if (this.rnd() > this.braid) continue;
            if (this.exits(e.c, e.r).length !== 1) continue;   // already fixed by a neighbour
            const shut = DIRS.filter((d) => {
                if (!this.hasWall(e.c, e.r, d)) return false;
                const nc = e.c + DC[d], nr = e.r + DR[d];
                return this.inside(nc, nr) && this.carved[this.idx(nc, nr)];
            });
            if (shut.length) this.carve(e.c, e.r, pick(this.rnd, shut));
        }
    }

    // ---------------------------------------------------------------- checks
    /** Flood fill from a carved cell; returns the reached set. */
    flood(start) {
        const seen = new Set([this.idx(start.c, start.r)]);
        const q = [start];
        while (q.length) {
            const { c, r } = q.pop();
            for (const d of this.exits(c, r)) {
                const nc = c + DC[d], nr = r + DR[d];
                if (!this.inside(nc, nr)) continue;
                const ni = this.idx(nc, nr);
                if (seen.has(ni)) continue;
                seen.add(ni);
                q.push({ c: nc, r: nr });
            }
        }
        return seen;
    }

    /** Every carved cell must be reachable from every other one. */
    connectivity() {
        let first = null, total = 0;
        for (let r = 0; r < this.rows; r++)
            for (let c = 0; c < this.cols; c++)
                if (this.carved[this.idx(c, r)]) { total++; if (!first) first = { c, r }; }
        if (!first) return { ok: false, reached: 0, total: 0 };
        const seen = this.flood(first);
        return { ok: seen.size === total, reached: seen.size, total };
    }

    /** Join any stragglers to the main body rather than leaving pockets. */
    repairConnectivity() {
        for (let guard = 0; guard < 64; guard++) {
            const st = this.connectivity();
            if (st.ok) return st;
            let first = null;
            for (let r = 0; r < this.rows && !first; r++)
                for (let c = 0; c < this.cols; c++)
                    if (this.carved[this.idx(c, r)]) { first = { c, r }; break; }
            const main = this.flood(first);
            // find an orphan touching the main body and knock that wall through
            let joined = false;
            for (let r = 0; r < this.rows && !joined; r++) {
                for (let c = 0; c < this.cols && !joined; c++) {
                    const i = this.idx(c, r);
                    if (!this.carved[i] || main.has(i)) continue;
                    for (const d of DIRS) {
                        const nc = c + DC[d], nr = r + DR[d];
                        if (!this.inside(nc, nr)) continue;
                        if (main.has(this.idx(nc, nr))) { this.carve(c, r, d); joined = true; break; }
                    }
                }
            }
            if (!joined) return this.connectivity();
        }
        return this.connectivity();
    }

    // ---------------------------------------------------------------- classify
    classify() {
        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                const i = this.idx(c, r);
                if (!this.carved[i]) { this.kind[i] = "solid"; continue; }
                if (this.room[i] >= 0) { this.kind[i] = "room"; continue; }
                const ex = this.exits(c, r);
                if (ex.length === 1) this.kind[i] = "dead-end";
                else if (ex.length >= 3) this.kind[i] = "junction";
                else if (ex.length === 2 && (ex[0] + 2) % 4 !== ex[1]) this.kind[i] = "corner";
                else this.kind[i] = "corridor";
            }
        }
    }

    // ---------------------------------------------------------------- torches
    /**
     * Wall brackets, on solid wall faces, spaced out so the labyrinth stays dark
     * between them. Dead ends and junctions get priority: they are where you stop
     * and look around. Also scatters unlit spare torches on room floors.
     */
    placeTorches() {
        const S = this.cell;
        const score = { "dead-end": 3, junction: 3, room: 2, corner: 1, corridor: 1 };
        const cands = [];
        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                const i = this.idx(c, r);
                if (!this.carved[i]) continue;
                for (const d of DIRS) {
                    if (!this.hasWall(c, r, d)) continue;  // bracket needs a solid face
                    cands.push({ c, r, d, w: (score[this.kind[i]] ?? 1) + this.rnd() });
                }
            }
        }
        cands.sort((a, b) => b.w - a.w);
        const taken = [];
        const minD = this.torchSpacing * S;
        for (const cd of cands) {
            // bracket sits on the inner face of the wall, at head height
            const cx = (cd.c + 0.5) * S, cz = (cd.r + 0.5) * S;
            const off = S / 2 - this.wallT * 0.5 - 0.06;
            const x = cx + DC[cd.d] * off, z = cz + DR[cd.d] * off;
            let ok = true;
            for (const t of taken) {
                if ((t.x - x) ** 2 + (t.z - z) ** 2 < minD * minD) { ok = false; break; }
            }
            if (!ok) continue;
            const t = {
                id: taken.length, c: cd.c, r: cd.r, side: cd.d,
                // above head height: the prop hangs from y-0.52, and the body's head
                // sphere reaches 1.96, so this keeps brackets solid without narrowing
                // the corridor the player has to walk down.
                x, y: this.wallH * 0.80, z,
                // inward normal: away from the wall it hangs on
                nx: -DC[cd.d], nz: -DR[cd.d],
                fuel: 0.55 + this.rnd() * 0.45,
                lit: true, present: true,
                phase: this.rnd() * Math.PI * 2,
            };
            taken.push(t);
        }
        this.torches = taken;

        // spare unlit torches lying in rooms, as replacements once yours burns out
        this.spares = [];
        for (const room of this.rooms) {
            if (this.rnd() < 0.35) continue;
            const c = room.c0 + Math.floor(this.rnd() * room.w);
            const r = room.r0 + Math.floor(this.rnd() * room.h);
            this.spares.push({
                id: this.spares.length, c, r,
                x: (c + 0.5) * S + (this.rnd() - 0.5) * S * 0.4,
                y: 0.10,
                z: (r + 0.5) * S + (this.rnd() - 0.5) * S * 0.4,
                fuel: 1.0, lit: false, present: true,
                phase: this.rnd() * Math.PI * 2,
            });
        }
    }

    // ---------------------------------------------------------------- spawn
    /** BFS distance in cells from a start cell. */
    distances(start) {
        const dist = new Int32Array(this.cols * this.rows).fill(-1);
        dist[this.idx(start.c, start.r)] = 0;
        const q = [start];
        for (let head = 0; head < q.length; head++) {
            const { c, r } = q[head];
            const dc = dist[this.idx(c, r)];
            for (const d of this.exits(c, r)) {
                const nc = c + DC[d], nr = r + DR[d];
                if (!this.inside(nc, nr)) continue;
                const ni = this.idx(nc, nr);
                if (dist[ni] >= 0) continue;
                dist[ni] = dc + 1;
                q.push({ c: nc, r: nr });
            }
        }
        return dist;
    }

    chooseSpawnAndGoal() {
        let start = null;
        for (let r = 0; r < this.rows && !start; r++)
            for (let c = 0; c < this.cols; c++)
                if (this.carved[this.idx(c, r)]) { start = { c, r }; break; }
        // double sweep: farthest from an arbitrary cell, then farthest from that
        const d1 = this.distances(start);
        let best = 0, bi = this.idx(start.c, start.r);
        for (let i = 0; i < d1.length; i++) if (d1[i] > best) { best = d1[i]; bi = i; }
        this.spawn = { c: bi % this.cols, r: Math.floor(bi / this.cols) };
        const d2 = this.distances(this.spawn);
        best = 0; bi = this.idx(this.spawn.c, this.spawn.r);
        for (let i = 0; i < d2.length; i++) if (d2[i] > best) { best = d2[i]; bi = i; }
        this.goal = { c: bi % this.cols, r: Math.floor(bi / this.cols) };
        this.goalDist = best;
    }

    generate() {
        this.carveRooms();
        this.growMaze();
        this.connectRooms();
        this.braidDeadEnds();
        this.repairConnectivity();
        this.classify();
        this.placeTorches();
        this.chooseSpawnAndGoal();
    }

    /** World-space centre of a cell. */
    centre(c, r) { return [(c + 0.5) * this.cell, 0, (r + 0.5) * this.cell]; }

    cellAt(x, z) {
        return { c: Math.floor(x / this.cell), r: Math.floor(z / this.cell) };
    }

    /**
     * Wall slabs near a world point, as AABBs, for collision.
     *
     * The cell size is deliberately NOT called S here: the module-level S is the
     * south direction constant, and shadowing it turned hasWall(c, r, S) into
     * hasWall(c, r, 3.0) — the west bit — which put phantom slabs across open
     * corridors and left the level's south boundary uncollidable.
     */
    wallsNear(x, z, span = 1) {
        const size = this.cell, T = this.wallT, out = [];
        const { c: c0, r: r0 } = this.cellAt(x, z);
        for (let r = r0 - span; r <= r0 + span; r++) {
            for (let c = c0 - span; c <= c0 + span; c++) {
                if (!this.inside(c, r)) continue;
                const i = this.idx(c, r);
                const x0 = c * size, z0 = r * size;
                if (!this.carved[i]) {              // solid rock: whole cell blocks
                    out.push({ x0, z0, x1: x0 + size, z1: z0 + size });
                    continue;
                }
                if (this.hasWall(c, r, N)) out.push({ x0, z0: z0 - T / 2, x1: x0 + size, z1: z0 + T / 2 });
                if (this.hasWall(c, r, S)) out.push({ x0, z0: z0 + size - T / 2, x1: x0 + size, z1: z0 + size + T / 2 });
                if (this.hasWall(c, r, W)) out.push({ x0: x0 - T / 2, z0, x1: x0 + T / 2, z1: z0 + size });
                if (this.hasWall(c, r, E)) out.push({ x0: x0 + size - T / 2, z0, x1: x0 + size + T / 2, z1: z0 + size });
            }
        }
        return out;
    }

    stats() {
        const counts = {};
        for (let i = 0; i < this.kind.length; i++) counts[this.kind[i]] = (counts[this.kind[i]] ?? 0) + 1;
        return { counts, rooms: this.rooms.length, torches: this.torches.length,
                 spares: this.spares.length, goalDist: this.goalDist, ...this.connectivity() };
    }
}
