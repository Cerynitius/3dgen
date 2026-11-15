// Column-major 4x4 matrices, matching what WebGL expects from uniformMatrix4fv.
//
// Convention, fixed here once so nothing downstream has to guess:
//   world is Y-up, the camera looks down -Z at yaw 0, +X is screen right.
// Player.forward()/right() in controls.js are defined to agree with viewFromPlayer(),
// and selftest.js asserts that agreement by transforming them through the matrix.

export function identity() {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

export function multiply(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
        for (let r = 0; r < 4; r++) {
            o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1]
                         + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
        }
    }
    return o;
}

export function perspective(fovY, aspect, near, far) {
    const f = 1 / Math.tan(fovY / 2);
    const o = new Float32Array(16);
    o[0] = f / aspect; o[5] = f;
    o[10] = (far + near) / (near - far); o[11] = -1;
    o[14] = (2 * far * near) / (near - far);
    return o;
}

export function lookAt(eye, centre, up) {
    let fx = centre[0] - eye[0], fy = centre[1] - eye[1], fz = centre[2] - eye[2];
    let rl = 1 / (Math.hypot(fx, fy, fz) || 1);
    fx *= rl; fy *= rl; fz *= rl;
    let sx = fy * up[2] - fz * up[1];
    let sy = fz * up[0] - fx * up[2];
    let sz = fx * up[1] - fy * up[0];
    rl = 1 / (Math.hypot(sx, sy, sz) || 1);
    sx *= rl; sy *= rl; sz *= rl;
    const ux = sy * fz - sz * fy, uy = sz * fx - sx * fz, uz = sx * fy - sy * fx;
    return new Float32Array([
        sx, ux, -fx, 0,
        sy, uy, -fy, 0,
        sz, uz, -fz, 0,
        -(sx * eye[0] + sy * eye[1] + sz * eye[2]),
        -(ux * eye[0] + uy * eye[1] + uz * eye[2]),
        fx * eye[0] + fy * eye[1] + fz * eye[2],
        1,
    ]);
}

/** World-to-camera for a Player. */
export function viewFromPlayer(player) {
    const e = player.eye();
    const d = player.viewDir();
    return lookAt(e, [e[0] + d[0], e[1] + d[1], e[2] + d[2]], [0, 1, 0]);
}

export function translation(x, y, z) {
    const o = identity();
    o[12] = x; o[13] = y; o[14] = z;
    return o;
}

export function rotationY(a) {
    const c = Math.cos(a), s = Math.sin(a);
    const o = identity();
    o[0] = c; o[2] = -s; o[8] = s; o[10] = c;
    return o;
}

export function rotationZ(a) {
    const c = Math.cos(a), s = Math.sin(a);
    const o = identity();
    o[0] = c; o[1] = s;
    o[4] = -s; o[5] = c;
    return o;
}

export function rotationX(a) {
    const c = Math.cos(a), s = Math.sin(a);
    const o = identity();
    o[5] = c; o[6] = s;
    o[9] = -s; o[10] = c;
    return o;
}

/** Transform a direction (w = 0) by a matrix. */
export function transformDir(m, v) {
    return [
        m[0] * v[0] + m[4] * v[1] + m[8] * v[2],
        m[1] * v[0] + m[5] * v[1] + m[9] * v[2],
        m[2] * v[0] + m[6] * v[1] + m[10] * v[2],
    ];
}

/** Transform a point (w = 1) by a matrix. */
export function transformPoint(m, v) {
    return [
        m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12],
        m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13],
        m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14],
    ];
}
