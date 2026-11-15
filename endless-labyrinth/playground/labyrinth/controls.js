// First-person controller.
//
// The split here is the whole point, and it is enforced by labyrinth/selftest.js:
//
//   Player.update(dt, keys, collide)  -- may move the body. MUST NOT touch orientation.
//   Player.look(dx, dy)               -- may turn the head. MUST NOT touch position.
//
// No key is bound to looking. The arrow keys are deliberately aliased onto walking
// so that the old "arrows/Q/E rotate the camera" behaviour cannot come back: the
// only thing that can change yaw or pitch is pointer movement.

export const BINDINGS = {
    forward: ["KeyW", "ArrowUp"],
    back: ["KeyS", "ArrowDown"],
    left: ["KeyA", "ArrowLeft"],
    right: ["KeyD", "ArrowRight"],
    sprint: ["ShiftLeft", "ShiftRight"],
    interact: ["KeyE"],
    ignite: ["KeyF"],
    drop: ["KeyQ"],
};

// Every code that means "walk". Nothing in here may rotate the view.
export const WALK_CODES = [
    ...BINDINGS.forward, ...BINDINGS.back, ...BINDINGS.left, ...BINDINGS.right,
];

export class Player {
    constructor(opts = {}) {
        this.pos = opts.pos ? opts.pos.slice() : [0, 0, 0];   // ground position, y unused
        this.yaw = opts.yaw ?? 0;
        this.pitch = opts.pitch ?? 0;
        this.vel = [0, 0];
        this.radius = opts.radius ?? 0.34;
        this.eyeHeight = opts.eyeHeight ?? 1.68;
        this.walkSpeed = opts.walkSpeed ?? 3.1;
        this.speedScale = 1;               // endless.js drops this to 0.25 while the tile ahead is still generating
        this.sprintMul = opts.sprintMul ?? 1.85;
        this.accel = opts.accel ?? 34;
        this.damping = opts.damping ?? 12;
        this.lookSens = opts.lookSens ?? 0.0022;
        this.pitchLimit = opts.pitchLimit ?? 1.45;
        this.bobPhase = 0;
        this.bob = 0;
        this.speed = 0;
        this.stuck = false;              // inside solid with no free direction nearby
    }

    // Y-up, camera looks down -Z at yaw 0, +X is screen right. These three must
    // agree with mat4.viewFromPlayer(); selftest.js proves it by pushing them
    // through the actual view matrix, which is the only way to catch a strafe or
    // mouse-look inversion — the vectors are self-consistent either way.
    /** Unit forward on the ground plane. */
    forward() { return [Math.sin(this.yaw), 0, -Math.cos(this.yaw)]; }
    /** Unit right on the ground plane, consistent with forward(). */
    right() { return [Math.cos(this.yaw), 0, Math.sin(this.yaw)]; }
    /** Full view direction, including pitch. */
    viewDir() {
        const cp = Math.cos(this.pitch);
        return [Math.sin(this.yaw) * cp, Math.sin(this.pitch), -Math.cos(this.yaw) * cp];
    }
    eye() { return [this.pos[0], this.eyeHeight + this.bob, this.pos[2]]; }

    /**
     * Turn the head. Pointer deltas only — this is the one and only place where
     * yaw and pitch may change.
     */
    look(dx, dy) {
        this.yaw += dx * this.lookSens;
        this.pitch -= dy * this.lookSens;
        if (this.pitch > this.pitchLimit) this.pitch = this.pitchLimit;
        if (this.pitch < -this.pitchLimit) this.pitch = -this.pitchLimit;
        // keep yaw in [0, 2pi) so long sessions do not drift into big floats
        const TAU = Math.PI * 2;
        this.yaw = ((this.yaw % TAU) + TAU) % TAU;
    }

    /**
     * Advance the body. `keys` is any Set-like with .has(code). `collide(x, z, r)`
     * returns true when a body of radius r at (x,z) overlaps the level.
     */
    update(dt, keys, collide) {
        const held = (list) => list.some((c) => keys.has(c));
        let fwd = 0, str = 0;
        if (held(BINDINGS.forward)) fwd += 1;
        if (held(BINDINGS.back)) fwd -= 1;
        if (held(BINDINGS.right)) str += 1;
        if (held(BINDINGS.left)) str -= 1;

        const len = Math.hypot(fwd, str);
        let wishX = 0, wishZ = 0;
        if (len > 0) {
            fwd /= len; str /= len;                 // no diagonal speed bonus
            const f = this.forward(), r = this.right();
            wishX = f[0] * fwd + r[0] * str;
            wishZ = f[2] * fwd + r[2] * str;
        }

        const target = this.walkSpeed * (held(BINDINGS.sprint) ? this.sprintMul : 1)
            * (this.speedScale ?? 1);
        // Accelerate toward the wish velocity, damp when there is no input.
        //
        // The blend is 1 - exp(-k*dt), the exact solution of dv/dt = k*(target - v) over
        // the step, not the Euler step min(1, k*dt) that was here. Euler makes the whole
        // controller frame-rate dependent: from a standstill, half a second of W covered
        // 1.5622 m at 60 Hz and 1.5019 m at 144 Hz, a 3.9% difference in how far you go
        // for the same input. Terminal speed matched, so it read as the start of every
        // movement being snappier on a high-refresh screen. The exponential form is
        // exact for any dt and needs no clamp, since it is already in [0, 1).
        //
        // Fixing the velocity alone only got the gap from 3.9% to 0.9%, because the
        // POSITION was still integrated as a rectangle, v(t+dt)*dt. Both halves have a
        // closed form over the step -- v decays exponentially toward the target, so the
        // displacement is its integral, target*dt + (v0 - target)*(1 - e^-k*dt)/k -- and
        // using both makes the controller identical at every refresh rate to float
        // precision, rather than merely closer.
        const tx = wishX * target, tz = wishZ * target;
        const k = len > 0 ? this.accel : this.damping;
        const blend = 1 - Math.exp(-k * dt);
        const v0x = this.vel[0], v0z = this.vel[1];
        let dx = tx * dt + (v0x - tx) * blend / k;
        let dz = tz * dt + (v0z - tz) * blend / k;
        this.vel[0] += (tx - v0x) * blend;
        this.vel[1] += (tz - v0z) * blend;
        if (Math.hypot(this.vel[0], this.vel[1]) < 1e-4) { this.vel[0] = 0; this.vel[1] = 0; }

        // Escape hatch. Axis-separated resolution below stops movement into solid, but
        // it has no answer for a body that is ALREADY inside solid: every axis is
        // blocked in both directions, velocity is zeroed each frame, and the player is
        // entombed with no input that can recover. Nothing should put them there --
        // but a collider that swaps representation mid-walk, a spawn point in a wall, or
        // an occupancy window that arrives late all can, and "walk into a corner and the
        // game ends" is not a failure worth preserving. Step out along the shortest free
        // direction instead. If everything nearby is solid too -- which is what the
        // endless client reports for cells whose wall bits have not loaded yet -- do
        // nothing rather than teleport into the unknown, and say so via `stuck`.
        this.stuck = false;
        if (collide && collide(this.pos[0], this.pos[2], this.radius)) {
            this.vel[0] = 0; this.vel[1] = 0;
            const step = this.radius * 0.5;
            let freed = false;
            for (let ring = 1; ring <= 12 && !freed; ring++)
                for (let a = 0; a < 16; a++) {
                    const th = (a / 16) * Math.PI * 2, d = ring * step;
                    const nx = this.pos[0] + Math.cos(th) * d;
                    const nz = this.pos[2] + Math.sin(th) * d;
                    if (collide(nx, nz, this.radius)) continue;
                    this.pos[0] = nx; this.pos[2] = nz; freed = true; break;
                }
            this.stuck = !freed;
            this.speed = 0;
            return 0;
        }

        // resolve axes separately so the body slides along a wall instead of sticking
        if (collide) {
            if (!collide(this.pos[0] + dx, this.pos[2], this.radius)) this.pos[0] += dx;
            else this.vel[0] = 0;
            if (!collide(this.pos[0], this.pos[2] + dz, this.radius)) this.pos[2] += dz;
            else this.vel[1] = 0;
        } else {
            this.pos[0] += dx;
            this.pos[2] += dz;
        }

        this.speed = Math.hypot(this.vel[0], this.vel[1]);
        this.bobPhase += this.speed * dt * 2.6;
        const amp = Math.min(this.speed / this.walkSpeed, 1.4) * 0.055;
        this.bob = Math.sin(this.bobPhase * 2) * amp;
        return this.speed;
    }
}

/** Circle-vs-AABB overlap, used by the level collider. */
export function circleHitsBox(x, z, r, b) {
    const nx = Math.max(b.x0, Math.min(x, b.x1));
    const nz = Math.max(b.z0, Math.min(z, b.z1));
    const dx = x - nx, dz = z - nz;
    return dx * dx + dz * dz < r * r;
}

/** Collider bound to a Labyrinth instance. */
export function makeCollider(maze) {
    return (x, z, r) => {
        for (const b of maze.wallsNear(x, z)) if (circleHitsBox(x, z, r, b)) return true;
        return false;
    };
}

/**
 * DOM wiring. Kept out of the classes above so the movement/look split stays
 * testable without a browser.
 */
export function bindInput(canvas, player, opts = {}) {
    const keys = new Set();
    const state = { locked: false, pressed: new Set() };
    const onPress = opts.onPress || (() => {});

    const down = (e) => {
        // never let the browser scroll the page out from under the canvas
        if (WALK_CODES.includes(e.code) || e.code === "Space") e.preventDefault();
        if (!keys.has(e.code)) {
            keys.add(e.code);
            state.pressed.add(e.code);
            onPress(e.code);
        }
    };
    const up = (e) => keys.delete(e.code);

    window.addEventListener("keydown", down, { passive: false });
    window.addEventListener("keyup", up);
    window.addEventListener("blur", () => keys.clear());

    canvas.addEventListener("mousedown", () => {
        if (document.pointerLockElement !== canvas) canvas.requestPointerLock();
    });
    document.addEventListener("pointerlockchange", () => {
        state.locked = document.pointerLockElement === canvas;
        if (!state.locked) keys.clear();
        if (opts.onLockChange) opts.onLockChange(state.locked);
    });
    document.addEventListener("mousemove", (e) => {
        if (document.pointerLockElement !== canvas) return;
        player.look(e.movementX, e.movementY);
    });
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    return { keys, state };
}
