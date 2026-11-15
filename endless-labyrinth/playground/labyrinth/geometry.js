// Mesh construction. Everything is plain triangle soup in one interleaved format:
//
//   [ px py pz | nx ny nz | u v | surface ]   9 floats per vertex
//
// `surface` selects the procedural material in the fragment shader:
//   0 floor, 1 wall, 2 ceiling, 3 wood, 4 metal, 5 skin.
//
// Wall faces are only emitted for the side that a carved cell can actually see, so
// the inside of solid rock is never triangulated.

import * as m4 from "./mat4.js";

export const FLOOR = 0, WALL = 1, CEIL = 2, WOOD = 3, METAL = 4, SKIN = 5;
export const STRIDE = 9;

// Prop placement lives here, and BOTH the renderer and the collider use it. When the
// renderer owned these matrices privately, the collider never saw the torch brackets
// and the player walked bodily through every one of them — with the camera ending up
// inside the shaft, closer than the near plane, so the prop vanished from the frame.
export function bracketModel(t) {
    return m4.multiply(m4.translation(t.x, t.y - 0.52, t.z),
                       m4.rotationY(Math.atan2(t.nx, t.nz)));
}
export function spareModel(s) {
    return m4.multiply(m4.translation(s.x, s.y, s.z), m4.rotationY(s.phase));
}

/** Apply a model matrix to an interleaved mesh, returning world-space triangles. */
export function transformMesh(mesh, model) {
    const out = new Float32Array(mesh.length);
    for (let i = 0; i < mesh.length; i += STRIDE) {
        const p = m4.transformPoint(model, [mesh[i], mesh[i + 1], mesh[i + 2]]);
        const n = m4.transformDir(model, [mesh[i + 3], mesh[i + 4], mesh[i + 5]]);
        out[i] = p[0]; out[i + 1] = p[1]; out[i + 2] = p[2];
        out[i + 3] = n[0]; out[i + 4] = n[1]; out[i + 5] = n[2];
        out[i + 6] = mesh[i + 6]; out[i + 7] = mesh[i + 7]; out[i + 8] = mesh[i + 8];
    }
    return out;
}

/**
 * Everything solid the player can run into: the level plus every prop currently in
 * the world, in world space. Rebuild this when a torch is taken or mounted.
 *
 * The level part is buildSolidLevelMesh, NOT buildLevelMesh. buildLevelMesh draws each
 * wall as its single inward-facing plane, which is right for rendering and wrong for
 * collision: two such planes face each other across the 2 m interior of the slab, and
 * nothing caps the slab where it ends at a T-junction or a dead end, so a body walks in
 * through the open end and is then inside the wall with a face on either side of it.
 * A function named "collision mesh" that returns a shell is a trap, so this one does not.
 */
export function buildCollisionMesh(maze) {
    const parts = [buildSolidLevelMesh(maze)];
    const torchMesh = buildTorchMesh();
    const looseMesh = buildLooseTorchMesh();
    for (const t of maze.torches) {
        if (t.present) parts.push(transformMesh(torchMesh, bracketModel(t)));
    }
    for (const s of maze.spares) {
        if (s.present) parts.push(transformMesh(looseMesh, spareModel(s)));
    }
    let n = 0;
    for (const p of parts) n += p.length;
    const all = new Float32Array(n);
    let o = 0;
    for (const p of parts) { all.set(p, o); o += p.length; }
    return all;
}

class MeshBuilder {
    constructor() { this.v = []; }
    /**
     * Quad from four corners. The winding is derived from the declared normal
     * rather than trusted from the caller: backface culling keys off winding in
     * window space, so a quad listed the wrong way round silently vanishes. This
     * reverses the corner order whenever the geometric normal would disagree with
     * `n`, which makes every face in the file correct by construction.
     */
    quad(a, b, c, d, n, uv, surf) {
        const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
        const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
        const gx = uy * vz - uz * vy, gy = uz * vx - ux * vz, gz = ux * vy - uy * vx;
        if (gx * n[0] + gy * n[1] + gz * n[2] < 0) {
            [a, b, c, d] = [d, c, b, a];
            uv = [uv[3], uv[2], uv[1], uv[0]];
        }
        const push = (p, u2) => this.v.push(p[0], p[1], p[2], n[0], n[1], n[2], u2[0], u2[1], surf);
        push(a, uv[0]); push(b, uv[1]); push(c, uv[2]);
        push(a, uv[0]); push(c, uv[2]); push(d, uv[3]);
    }
    /** Axis-aligned box, used for torch shafts, brackets and hands. */
    box(cx, cy, cz, sx, sy, sz, surf) {
        const x0 = cx - sx / 2, x1 = cx + sx / 2;
        const y0 = cy - sy / 2, y1 = cy + sy / 2;
        const z0 = cz - sz / 2, z1 = cz + sz / 2;
        const uv = [[0, 0], [1, 0], [1, 1], [0, 1]];
        this.quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, 1], uv, surf);
        this.quad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [0, 0, -1], uv, surf);
        this.quad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [1, 0, 0], uv, surf);
        this.quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [-1, 0, 0], uv, surf);
        this.quad([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], [0, 1, 0], uv, surf);
        this.quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], [0, -1, 0], uv, surf);
    }
    data() { return new Float32Array(this.v); }
    get count() { return this.v.length / STRIDE; }
}

/**
 * The level: floor and ceiling per carved cell, plus the inward face of every wall
 * that a carved cell touches.
 *
 * RENDER ONLY. Every wall here is one zero-thickness plane pushed T/2 into the cell, so
 * the mesh is a shell: the interior of each slab is empty space bounded by two planes
 * that face away from each other, and a slab that ends at a junction is not capped at
 * all. Colliding against it lets a body walk into a wall through the open end and get
 * shut inside. Use buildSolidLevelMesh (or maze.wallsNear, which returns the slabs
 * directly as AABBs) for anything that has to stop a body.
 */
export function buildLevelMesh(maze) {
    const mb = new MeshBuilder();
    const S = maze.cell, T = maze.wallT, H = maze.wallH;

    for (let r = 0; r < maze.rows; r++) {
        for (let c = 0; c < maze.cols; c++) {
            if (!maze.carved[maze.idx(c, r)]) continue;
            const x0 = c * S, z0 = r * S, x1 = x0 + S, z1 = z0 + S;

            // floor
            mb.quad([x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1], [0, 1, 0],
                    [[x0, z0], [x1, z0], [x1, z1], [x0, z1]], FLOOR);
            // ceiling (normal points down, into the corridor)
            mb.quad([x0, H, z1], [x1, H, z1], [x1, H, z0], [x0, H, z0], [0, -1, 0],
                    [[x0, z1], [x1, z1], [x1, z0], [x0, z0]], CEIL);

            // north wall: plane at z0 + T/2, facing +z
            if (maze.hasWall(c, r, 0)) {
                const z = z0 + T / 2;
                mb.quad([x0, 0, z], [x1, 0, z], [x1, H, z], [x0, H, z], [0, 0, 1],
                        [[x0, 0], [x1, 0], [x1, H], [x0, H]], WALL);
            }
            // south wall: plane at z1 - T/2, facing -z
            if (maze.hasWall(c, r, 2)) {
                const z = z1 - T / 2;
                mb.quad([x1, 0, z], [x0, 0, z], [x0, H, z], [x1, H, z], [0, 0, -1],
                        [[x1, 0], [x0, 0], [x0, H], [x1, H]], WALL);
            }
            // west wall: plane at x0 + T/2, facing +x
            if (maze.hasWall(c, r, 3)) {
                const x = x0 + T / 2;
                mb.quad([x, 0, z1], [x, 0, z0], [x, H, z0], [x, H, z1], [1, 0, 0],
                        [[z1, 0], [z0, 0], [z0, H], [z1, H]], WALL);
            }
            // east wall: plane at x1 - T/2, facing -x
            if (maze.hasWall(c, r, 1)) {
                const x = x1 - T / 2;
                mb.quad([x, 0, z0], [x, 0, z1], [x, H, z1], [x, H, z0], [-1, 0, 0],
                        [[z0, 0], [z1, 0], [z1, H], [z0, H]], WALL);
            }
        }
    }
    return mb.data();
}

/**
 * The level as closed volumes: every wall slab as a sealed box, plus floor and ceiling.
 *
 * This is the collision counterpart of buildLevelMesh. The slabs are the same ones
 * maze.wallsNear returns as AABBs, extruded to the ceiling, so the two colliders agree
 * by construction. Neighbouring cells name the shared slab twice -- cell (c,r)'s north
 * wall is cell (c,r-1)'s south wall -- so boxes are emitted once, keyed on their bounds.
 */
export function buildSolidLevelMesh(maze) {
    const mb = new MeshBuilder();
    const S = maze.cell, T = maze.wallT, H = maze.wallH;
    const seen = new Set();
    const slab = (x0, z0, x1, z1) => {
        const k = `${x0},${z0},${x1},${z1}`;
        if (seen.has(k)) return;
        seen.add(k);
        mb.box((x0 + x1) / 2, H / 2, (z0 + z1) / 2, x1 - x0, H, z1 - z0, WALL);
    };

    for (let r = 0; r < maze.rows; r++) {
        for (let c = 0; c < maze.cols; c++) {
            const x0 = c * S, z0 = r * S, x1 = x0 + S, z1 = z0 + S;
            if (!maze.carved[maze.idx(c, r)]) {      // solid rock: the whole cell blocks
                slab(x0, z0, x1, z1);
                continue;
            }
            mb.quad([x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1], [0, 1, 0],
                    [[x0, z0], [x1, z0], [x1, z1], [x0, z1]], FLOOR);
            mb.quad([x0, H, z1], [x1, H, z1], [x1, H, z0], [x0, H, z0], [0, -1, 0],
                    [[x0, z1], [x1, z1], [x1, z0], [x0, z0]], CEIL);
            if (maze.hasWall(c, r, 0)) slab(x0, z0 - T / 2, x1, z0 + T / 2);
            if (maze.hasWall(c, r, 2)) slab(x0, z1 - T / 2, x1, z1 + T / 2);
            if (maze.hasWall(c, r, 3)) slab(x0 - T / 2, z0, x0 + T / 2, z1);
            if (maze.hasWall(c, r, 1)) slab(x1 - T / 2, z0, x1 + T / 2, z1);
        }
    }
    return mb.data();
}

/**
 * A torch prop in local space: shaft along +y, head at the top, plus the iron
 * bracket ring. Local origin sits where the bracket meets the wall.
 */
export function buildTorchMesh() {
    const mb = new MeshBuilder();
    mb.box(0, 0, 0.06, 0.05, 0.05, 0.12, METAL);      // wall plate
    mb.box(0, 0.05, 0.16, 0.04, 0.04, 0.22, METAL);   // arm out of the wall
    mb.box(0, 0.14, 0.26, 0.10, 0.10, 0.10, METAL);   // ring
    mb.box(0, 0.30, 0.26, 0.055, 0.42, 0.055, WOOD);  // shaft
    mb.box(0, 0.52, 0.26, 0.085, 0.10, 0.085, WOOD);  // pitch-soaked head
    return mb.data();
}

/** A loose torch lying on the floor. */
export function buildLooseTorchMesh() {
    const mb = new MeshBuilder();
    mb.box(0, 0.04, 0, 0.05, 0.05, 0.5, WOOD);
    mb.box(0, 0.04, 0.28, 0.08, 0.08, 0.10, WOOD);
    return mb.data();
}

/**
 * The first-person hand holding a torch, in view space: +x right, +y up, and the
 * camera looks down -z, so the arm comes in from the lower right.
 */
export function buildHandMesh() {
    const mb = new MeshBuilder();
    // forearm angling in from bottom-right
    mb.box(0.26, -0.30, -0.62, 0.10, 0.10, 0.34, SKIN);
    mb.box(0.24, -0.22, -0.78, 0.11, 0.12, 0.12, SKIN);   // fist
    mb.box(0.24, -0.05, -0.80, 0.05, 0.40, 0.05, WOOD);   // torch shaft
    mb.box(0.24, 0.17, -0.80, 0.085, 0.10, 0.085, WOOD);  // head
    return mb.data();
}

/** A unit quad on the XY plane, expanded into a camera-facing billboard. */
export function buildBillboard() {
    return new Float32Array([
        -1, -1, 0, 0, 0, 1, 0, 0, 0,
         1, -1, 0, 0, 0, 1, 1, 0, 0,
         1,  1, 0, 0, 0, 1, 1, 1, 0,
        -1, -1, 0, 0, 0, 1, 0, 0, 0,
         1,  1, 0, 0, 0, 1, 1, 1, 0,
        -1,  1, 0, 0, 0, 1, 0, 1, 0,
    ]);
}

/**
 * Line of sight across the cell grid, so a torch two corridors away does not
 * glow through the rock in front of you. Amanatides-Woo DDA, stepping cell to
 * cell and refusing to cross a wall.
 */
export function losClear(maze, x0, z0, x1, z1) {
    const S = maze.cell;
    let c = Math.floor(x0 / S), r = Math.floor(z0 / S);
    const cEnd = Math.floor(x1 / S), rEnd = Math.floor(z1 / S);
    if (c === cEnd && r === rEnd) return true;
    const dx = x1 - x0, dz = z1 - z0;
    const stepC = Math.sign(dx), stepR = Math.sign(dz);
    const invX = dx !== 0 ? 1 / Math.abs(dx) : Infinity;
    const invZ = dz !== 0 ? 1 / Math.abs(dz) : Infinity;
    let tMaxX = dx !== 0
        ? ((stepC > 0 ? (c + 1) * S - x0 : x0 - c * S) / Math.abs(dx)) : Infinity;
    let tMaxZ = dz !== 0
        ? ((stepR > 0 ? (r + 1) * S - z0 : z0 - r * S) / Math.abs(dz)) : Infinity;
    const tDeltaX = S * invX, tDeltaZ = S * invZ;

    for (let guard = 0; guard < 256; guard++) {
        let side;
        if (tMaxX < tMaxZ) { tMaxX += tDeltaX; c += stepC; side = stepC > 0 ? 1 : 3; }
        else { tMaxZ += tDeltaZ; r += stepR; side = stepR > 0 ? 2 : 0; }
        if (!maze.inside(c, r)) return false;
        // we just crossed into (c,r) through the opposite of `side` on the old cell
        const prevC = c - (side === 1 ? 1 : side === 3 ? -1 : 0);
        const prevR = r - (side === 2 ? 1 : side === 0 ? -1 : 0);
        if (maze.inside(prevC, prevR) && maze.hasWall(prevC, prevR, side)) return false;
        if (!maze.carved[maze.idx(c, r)]) return false;
        if (c === cEnd && r === rEnd) return true;
        if (tMaxX > 4 && tMaxZ > 4) return true;   // ran past the segment
    }
    return true;
}
