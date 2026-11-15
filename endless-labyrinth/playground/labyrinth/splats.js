// Real-time Gaussian splat renderer for chunks streamed from chunk_server.py.
//
// Wire format, little-endian float32, 14 per splat:
//   [ x y z | sx sy sz | qw qx qy qz | r g b | a ]
// The quaternion is SCALAR-FIRST. This comment said `qx qy qz qw` while the vertex
// shader thirty lines down said the opposite and unpacked scalar-first; one of the two
// had to be wrong, and it was this one. The server dumps the model's rotation head
// straight onto the wire, the official exporter writes that same head into rot_0..rot_3
// of a 3DGS PLY, and rot_0 is the scalar -- which is also what recon.calibrate_convention
// measures when it picks "wxyz" over "xyzw" against a held-out raycast.
// Conventions verified against the server's own output rather than assumed:
//   scales   linear metres (the offline PLY writer applies .log(), the tensors do not)
//   quats    already unit length
//   opacity  already in [0,1] (no sigmoid needed)
//   colors   SH DC coefficients, range about +-2  ->  rgb = 0.5 + 0.28209*f_dc
//
// Splats live in one growing RGBA32F texture (4 texels each) so chunks can be appended
// without rebuilding anything. Correct alpha compositing needs back-to-front order, so
// a worker re-sorts by view depth off the main thread and hands back an index buffer.

const TEX_W = 2048;              // texels; 4 per splat, so 512 splats per row
const FLOATS = 14;

const SORT_WORKER = `
let pos = null;      // Float32Array xyz per splat, indexed by global slot position
let slots = [];      // [start, count] of each LIVE slot; evicted slots are excluded
let view = null;
let gen = 0;         // slot-layout generation this sorter is working from
self.onmessage = (e) => {
    const d = e.data;
    // Accumulate whatever arrived. An earlier version returned as soon as it saw
    // positions, so the slot list that came in the SAME message was never read: the
    // sorter had nothing live, drew zero splats, and the window was black.
    if (d.capacity) { pos = new Float32Array(d.capacity * 3); return; }
    // Only the slot that changed is sent, and its buffer is transferred rather than
    // copied. Posting the whole position array instead structured-cloned 91.5 MB on
    // every single tile -- about 42 ms of main-thread stall each, ten times per tile
    // crossing -- to deliver the few hundred thousand positions that had actually moved.
    if (d.chunk) pos.set(d.chunk, d.base);
    if (d.slots) slots = d.slots;
    if (d.view) view = d.view;
    if (d.gen !== undefined) gen = d.gen;
    if (!pos || !slots.length || !view) return;
    let count = 0;
    for (let i = 0; i < slots.length; i += 2) count += slots[i + 1];
    if (!count) return;
    const idxOf = new Uint32Array(count);
    let w = 0;
    for (let i = 0; i < slots.length; i += 2)
        for (let k = 0; k < slots[i + 1]; k++) idxOf[w++] = slots[i] + k;
    const v = view;                        // world->cam, column major
    // depth along the view direction; only the z row matters
    const depths = new Int32Array(count);
    let dmin = Infinity, dmax = -Infinity;
    for (let n = 0; n < count; n++) {
        const i = idxOf[n];
        const x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
        const dz = v[2]*x + v[6]*y + v[10]*z + v[14];
        const q = dz * 4096 | 0;
        depths[n] = q;
        if (q < dmin) dmin = q;
        if (q > dmax) dmax = q;
    }
    // counting sort, far to near (more negative z = further in front of the camera)
    const range = (dmax - dmin) || 1;
    const BUCKETS = 256 * 256;
    const hist = new Uint32Array(BUCKETS + 1);
    const bucket = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
        const b = ((depths[i] - dmin) / range * (BUCKETS - 1)) | 0;
        bucket[i] = b; hist[b]++;
    }
    let acc = 0;
    for (let b = 0; b < BUCKETS; b++) { const c = hist[b]; hist[b] = acc; acc += c; }
    const order = new Uint32Array(count);
    for (let n = 0; n < count; n++) order[hist[bucket[n]]++] = idxOf[n];
    self.postMessage({ order, count, gen }, [order.buffer]);
};`;

export const VS = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_splats;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec2 u_focal;
uniform vec2 u_viewport;
uniform float u_gain;      // opacity compensation for a sparse field
uniform float u_dilate;    // screen-space minimum footprint, px^2
in vec2 a_corner;
in uint a_index;
out vec4 v_colour;
out vec3 v_conic;
out vec2 v_delta;
out float v_depth;

vec4 fetch(uint i, uint slot) {
    uint t = i * 4u + slot;
    return texelFetch(u_splats, ivec2(int(t % ${TEX_W}u), int(t / ${TEX_W}u)), 0);
}

void main() {
    vec4 a = fetch(a_index, 0u);            // x y z sx
    vec4 b = fetch(a_index, 1u);            // sy sz qx qy
    vec4 c = fetch(a_index, 2u);            // qz qw r g
    vec4 d = fetch(a_index, 3u);            // bl alpha _ _

    vec3 centre = a.xyz;
    vec3 scale = vec3(a.w, b.x, b.y);
    // The wire order is w x y z, not x y z w. The model's rotation head is written
    // straight into rot_0..rot_3 of a 3DGS PLY by the official exporter, and rot_0 is
    // the scalar there -- so reading it as xyzw rotates every splat by a quarter turn
    // about a mixed axis. It went unnoticed because these splats are nearly isotropic,
    // which makes the error cheap rather than absent.
    vec4 q = vec4(b.w, c.x, c.y, b.z);      // x y z w, from wire w x y z
    vec3 sh = vec3(c.z, c.w, d.x);
    float alpha = d.y;

    vec4 cam = u_view * vec4(centre, 1.0);
    if (cam.z > -0.05) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    v_depth = -cam.z;

    // 3D covariance from rotation and scale
    mat3 R = mat3(
        1.0 - 2.0*(q.y*q.y + q.z*q.z), 2.0*(q.x*q.y + q.z*q.w),       2.0*(q.x*q.z - q.y*q.w),
        2.0*(q.x*q.y - q.z*q.w),       1.0 - 2.0*(q.x*q.x + q.z*q.z), 2.0*(q.y*q.z + q.x*q.w),
        2.0*(q.x*q.z + q.y*q.w),       2.0*(q.y*q.z - q.x*q.w),       1.0 - 2.0*(q.x*q.x + q.y*q.y));
    mat3 S = mat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
    mat3 M = R * S;
    mat3 Sigma = M * transpose(M);

    // project: J is the jacobian of the perspective divide at cam
    float z2 = cam.z * cam.z;
    mat3 J = mat3(
        u_focal.x / cam.z, 0.0, -(u_focal.x * cam.x) / z2,
        0.0, u_focal.y / cam.z, -(u_focal.y * cam.y) / z2,
        0.0, 0.0, 0.0);
    mat3 W = mat3(u_view);
    mat3 T = J * W;
    mat3 cov = T * Sigma * transpose(T);
    float cxx = cov[0][0] + u_dilate;       // dilation keeps thin splats > 1px
    float cxy = cov[0][1];
    float cyy = cov[1][1] + u_dilate;

    float det = cxx * cyy - cxy * cxy;
    if (det <= 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    v_conic = vec3(cyy, -cxy, cxx) / det;   // inverse 2x2, packed

    // eigenvalues give the ellipse axes; 3 sigma covers it
    float mid = 0.5 * (cxx + cyy);
    float rad = sqrt(max(mid * mid - det, 0.1));
    float l1 = mid + rad, l2 = max(mid - rad, 0.1);
    vec2 major = normalize(vec2(cxy, l1 - cxx) + vec2(1e-6, 0.0));
    vec2 minor = vec2(-major.y, major.x);
    vec2 offset = a_corner.x * major * 3.0 * sqrt(l1)
                + a_corner.y * minor * 3.0 * sqrt(l2);

    vec4 clip = u_proj * cam;
    gl_Position = vec4(clip.xy + offset / u_viewport * 2.0 * clip.w, clip.zw);
    v_delta = offset;
    // A pixel-aligned splat field is one Gaussian deep per view, while the per-splat
    // opacity (mean ~0.22 here) assumes many overlapping layers. Compositing it as-is
    // leaves 60-80% background showing: dark and see-through. This is the alpha that
    // u_gain stacked layers would have produced.
    float covered = clamp(1.0 - pow(max(1.0 - alpha, 0.0), u_gain), 0.0, 1.0);
    v_colour = vec4(max(vec3(0.5) + 0.28209479 * sh, 0.0), covered);

    // Warm golden grade over the flat-lit reconstruction: the model gets clean
    // equal-signal views, the game gets a sun-baked palette. No per-torch math --
    // just colour. Slight exposure lift so corridors do not read as grey concrete.
    v_colour.rgb = min(v_colour.rgb * vec3(1.30, 1.20, 1.02) + vec3(0.05, 0.035, 0.02), 1.0);
}`;

export const FS = `#version 300 es
precision highp float;
in vec4 v_colour;
in vec3 v_conic;
in vec2 v_delta;
in float v_depth;
uniform float u_fogDensity;   // 0 = clear air; endless.js cranks it while the tile ahead generates
uniform vec3 u_fogColor;
out vec4 frag;
void main() {
    float p = -0.5 * (v_conic.x * v_delta.x * v_delta.x
                    + v_conic.z * v_delta.y * v_delta.y)
              - v_conic.y * v_delta.x * v_delta.y;
    if (p > 0.0) discard;
    float a = v_colour.a * exp(p);
    if (a < 0.004) discard;
    // Height-blind distance fog, squared falloff so the near field stays readable.
    // Premultiplied output: blend the colour toward fog, keep the splat's own alpha.
    float fog = 1.0 - exp(-u_fogDensity * u_fogDensity * v_depth * v_depth);
    frag = vec4(mix(v_colour.rgb, u_fogColor, fog) * a, a);
}`;

function compile(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
        throw new Error("splat shader: " + gl.getShaderInfoLog(s));
    return s;
}

export class SplatField {
    constructor(gl, capacity = 4_000_000, slots = 24) {
        this.gl = gl;
        // one slot per cell; 4 texels per splat, and TEX_W/4 splats per texel row, so
        // rounding the slot to whole rows keeps every upload a single texSubImage2D
        const perRow = TEX_W / 4;
        this.slotSize = Math.floor(capacity / slots / perRow) * perRow;

        // The field lives in ONE texture, so its splat capacity is capped by
        // MAX_TEXTURE_SIZE: 4 texels per splat over a 2048-wide texture means
        // maxTexels/4 splats and no more. 8M splats needs 15,624 rows, and the WebGL2
        // floor is 8192 -- on any GPU sitting at that floor texStorage2D fails, nothing
        // checks, and the result is a black window with no error at all. When it does
        // not fit, drop SLOTS and never slot SIZE: a slot smaller than the largest tile
        // the server can emit truncates it, and truncating a voxel-pruned list is the
        // one thing pruning was chosen to avoid, since it removes coverage in patches
        // rather than uniformly.
        const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) || 8192;
        const maxSplats = Math.floor((maxTex * TEX_W) / 4);
        if (this.slotSize * slots > maxSplats) {
            const fit = Math.floor(maxSplats / this.slotSize);
            if (fit < 1)
                throw new Error(
                    `splat field: one slot needs ${this.slotSize.toLocaleString()} splats ` +
                    `(${Math.ceil(this.slotSize * 4 / TEX_W)} texture rows) but ` +
                    `MAX_TEXTURE_SIZE is ${maxTex}. Lower the server's max_splats budget.`);
            console.warn(
                `splat field: MAX_TEXTURE_SIZE ${maxTex} holds ${maxSplats.toLocaleString()} ` +
                `splats; ${slots} slots of ${this.slotSize.toLocaleString()} want ` +
                `${(this.slotSize * slots).toLocaleString()}. Using ${fit} slots.`);
            slots = fit;
        }
        this.slotCount = slots;
        capacity = this.slotSize * slots;
        this.generation = 0;
        this.fogDensity = 0.0;           // endless.js drives this toward 0.14 while content generates
        this.fogColor = [0.055, 0.045, 0.035];   // warm tomb-dark, not cold blue
        this.torchData = new Float32Array(24);   // 8 nearest torches, xyz each
        this.torchCount = 0;
        this.timeSec = 0;
        this.gl = gl;
        this.capacity = capacity;
        this.count = 0;
        this.slotOf = new Map();          // cell key -> slot
        this.live = new Map();            // slot -> splat count
        this.slotCost = new Float32Array(slots);
        this.free = Array.from({ length: slots }, (_, i) => slots - 1 - i);
        this.evictions = 0;
        // Coarse occupancy built from the splats themselves. Collision has to agree
        // with what is actually on screen: colliding against the procedural maze while
        // the visible world is a reconstruction means walking through what you can see
        // and stopping at what you cannot.
        this.VOX = 0.35;                  // metres
        this.occ = new Map();             // "vx,vz" -> splat count in the body band
        this.slotVox = new Map();         // slot -> [keys] so eviction can undo them
        // Positions live in the WORKER, not here: this side only ever posted them
        // across, and keeping a second 91.5 MB copy on the main heap bought nothing.
        this.staging = new Float32Array(TEX_W * 4);     // one row at a time
        this.sortedCount = 0;
        this.dirty = false;

        this.prog = gl.createProgram();
        gl.attachShader(this.prog, compile(gl, gl.VERTEX_SHADER, VS));
        gl.attachShader(this.prog, compile(gl, gl.FRAGMENT_SHADER, FS));
        gl.linkProgram(this.prog);
        if (!gl.getProgramParameter(this.prog, gl.LINK_STATUS))
            throw new Error("splat link: " + gl.getProgramInfoLog(this.prog));

        this.u = {};
        for (const n of ["u_splats", "u_view", "u_proj", "u_focal", "u_viewport", "u_gain", "u_dilate", "u_fogDensity", "u_fogColor"])
            this.u[n] = gl.getUniformLocation(this.prog, n);

        // texture: 4 texels per splat
        const rows = Math.ceil((capacity * 4) / TEX_W);
        this.texRows = rows;
        this.tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.tex);
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, TEX_W, rows);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);
        const quad = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, quad);
        gl.bufferData(gl.ARRAY_BUFFER,
            new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW);
        const cLoc = gl.getAttribLocation(this.prog, "a_corner");
        gl.enableVertexAttribArray(cLoc);
        gl.vertexAttribPointer(cLoc, 2, gl.FLOAT, false, 0, 0);

        this.indexBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
        gl.bufferData(gl.ARRAY_BUFFER, capacity * 4, gl.DYNAMIC_DRAW);
        const iLoc = gl.getAttribLocation(this.prog, "a_index");
        gl.enableVertexAttribArray(iLoc);
        gl.vertexAttribIPointer(iLoc, 1, gl.UNSIGNED_INT, 0, 0);
        gl.vertexAttribDivisor(iLoc, 1);
        gl.bindVertexArray(null);

        this.worker = new Worker(URL.createObjectURL(
            new Blob([SORT_WORKER], { type: "text/javascript" })));
        this.worker.onmessage = (e) => {
            const { order, count, gen } = e.data;
            this.sorting = false;
            // A sort takes a round trip, and a tile can arrive and evict a slot inside
            // it. The returned order then indexes rows that now belong to a different
            // tile -- and worse, the tail of a reused slot still holds the evicted
            // tile's bytes, because texSubImage2D only overwrites as many rows as the
            // new tile needs. Drawing that order paints the dead tile's splats, at the
            // wrong depth order, wherever it used to be: the flickering wall fragments.
            // The layout generation is stamped on the request and echoed on the reply,
            // so a stale one is dropped and the next frame simply asks again.
            if (gen !== this.generation) return;
            gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, order, 0, count);
            this.sortedCount = count;
            this.lastOrder = order;      // CPU copy so addChunk can filter a reused slot's indices out immediately
        };
        this.sorting = false;
        this.worker.postMessage({ capacity: this.capacity });
    }

    /**
     * Store a chunk in a slot, evicting the least useful one when full.
     *
     * This was append-only with a flat capacity, so after a handful of chunks the
     * field silently stopped accepting anything and the world simply stopped
     * appearing however far you walked. Fixed slots make the memory bound explicit
     * and let old cells be reclaimed as you move away from them.
     *
     * @param key    cell identity, so a re-fetch replaces rather than duplicates
     * @param cost   eviction score; the highest-cost live slot is dropped first
     */
    addChunk(key, data, transform = null, cost = 0) {
        const gl = this.gl;
        const n = Math.floor(data.length / FLOATS);
        const room = Math.min(n, this.slotSize);
        if (room <= 0) return 0;
        // Truncation is not a graceful degradation here. voxel_prune returns a list that
        // is spatially uniform, so dropping its tail removes whole regions of surface
        // rather than thinning everything -- patch-shaped holes in otherwise solid
        // walls. It used to happen silently; say so, loudly, once per tile.
        if (n > room)
            console.warn(`splat field: tile ${key} has ${n.toLocaleString()} splats but a ` +
                         `slot holds ${room.toLocaleString()}; ${(n - room).toLocaleString()} ` +
                         `dropped, which will show as patches missing from walls.`);

        let slot = this.slotOf.get(key);
        if (slot === undefined) {
            slot = this.free.pop();
            if (slot === undefined) {                     // evict the worst live slot
                let worst = -1, worstCost = -Infinity;
                for (const [k, sl] of this.slotOf) {
                    const c = this.slotCost[sl];
                    if (c > worstCost) { worstCost = c; worst = sl; this.worstKey = k; }
                }
                if (worst < 0 || worstCost <= cost) return 0;   // nothing worth evicting
                this.slotOf.delete(this.worstKey);
                this.evictions++;
                slot = worst;
            }
            this.slotOf.set(key, slot);
        }
        this.slotCost[slot] = cost;
        const base = slot * this.slotSize;

        const packed = new Float32Array(room * 16);
        const chunkPos = new Float32Array(room * 3);    // transferred to the sorter
        let tScale = 1;
        if (transform) tScale = Math.hypot(transform[0], transform[1], transform[2]) || 1;
        for (let i = 0; i < room; i++) {
            const s = i * FLOATS, o = i * 16;
            let x = data[s], y = data[s + 1], z = data[s + 2];
            if (transform) {
                const t = transform;
                const nx = t[0]*x + t[4]*y + t[8]*z + t[12];
                const ny = t[1]*x + t[5]*y + t[9]*z + t[13];
                const nz = t[2]*x + t[6]*y + t[10]*z + t[14];
                x = nx; y = ny; z = nz;
            }
            packed[o] = x; packed[o+1] = y; packed[o+2] = z;
            packed[o+3] = data[s+3] * tScale;
            packed[o+4] = data[s+4] * tScale;
            packed[o+5] = data[s+5] * tScale;
            packed[o+6] = data[s+6]; packed[o+7] = data[s+7];
            packed[o+8] = data[s+8]; packed[o+9] = data[s+9];
            packed[o+10] = data[s+10]; packed[o+11] = data[s+11];
            packed[o+12] = data[s+12]; packed[o+13] = data[s+13];
            packed[o+14] = 0; packed[o+15] = 0;
            const p = i * 3;
            chunkPos[p] = x; chunkPos[p+1] = y; chunkPos[p+2] = z;
        }

        // slot boundaries are texel-row aligned by construction, so this is one upload.
        // Upload the WHOLE slot, not just the new tile's rows: a reused slot's tail still
        // holds the previous tile's texels, and any order buffer already on the GPU keeps
        // drawing them as ghosts until the next sort reply lands. The tail is filled with
        // INERT splats, not zeros: alpha 0 makes them invisible, but a zeroed tail has
        // zero-norm quaternions and zero scales, and one stale index drawing them poisons
        // the whole frame to black (NaN). qw=1 + tiny scale keeps them harmless.
        const rows = Math.ceil((this.slotSize * 4) / TEX_W);
        const padded = new Float32Array(rows * TEX_W * 4);
        padded.set(packed);
        for (let t = room * 16; t < padded.length; t += 16) {
            padded[t + 3] = 1e-4; padded[t + 4] = 1e-4; padded[t + 5] = 1e-4;  // scale
            padded[t + 6] = 1;                                                  // qw = identity
            // alpha stays 0 -> never visible
        }
        gl.bindTexture(gl.TEXTURE_2D, this.tex);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, (base * 4) / TEX_W, TEX_W, rows,
                         gl.RGBA, gl.FLOAT, padded);

        // rebuild this slot's occupancy contribution
        const oldKeys = this.slotVox.get(slot);
        if (oldKeys) for (const k of oldKeys) {
            const n2 = (this.occ.get(k) || 0) - 1;
            if (n2 > 0) this.occ.set(k, n2); else this.occ.delete(k);
        }
        const keys = [];
        const seenVox = new Set();
        for (let i = 0; i < room; i++) {
            const o = i * 16;
            const y = packed[o + 1];
            if (y < 0.35 || y > 1.85) continue;          // only the band a body occupies
            const k = Math.floor(packed[o] / this.VOX) + "," + Math.floor(packed[o + 2] / this.VOX);
            if (seenVox.has(k)) continue;
            seenVox.add(k);
            keys.push(k);
            this.occ.set(k, (this.occ.get(k) || 0) + 1);
        }
        this.slotVox.set(slot, keys);
        this.live.set(slot, room);

        // The order buffer on the GPU still indexes this slot's previous tenant.
        // Filter those indices out NOW: the new tile is absent for the frames until
        // the next sort reply, instead of drawn at whatever depth the dead tile's
        // splats happened to sort to -- that wrong-depth frame is the load flicker.
        if (this.lastOrder) {
            const loIx = base, hiIx = base + this.slotSize;
            const keep = this.lastOrder.filter(ix => ix < loIx || ix >= hiIx);
            gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, keep, 0, keep.length);
            this.sortedCount = keep.length;
            this.lastOrder = keep;
        }
        this.count = 0;
        for (const c of this.live.values()) this.count += c;
        // The slot layout has changed, so any sort still in flight is now describing a
        // field that no longer exists. Bumping the generation is what lets its reply be
        // recognised as stale and dropped.
        this.generation++;
        this.pushSlots(base * 3, chunkPos);
        return room;
    }

    /** Is there enough reconstructed matter at (x,z) to stop a body of radius r? */
    blocked(x, z, r = 0.34) {
        const v = this.VOX;
        const x0 = Math.floor((x - r) / v), x1 = Math.floor((x + r) / v);
        const z0 = Math.floor((z - r) / v), z1 = Math.floor((z + r) / v);
        for (let vz = z0; vz <= z1; vz++)
            for (let vx = x0; vx <= x1; vx++)
                if (this.occ.has(vx + "," + vz)) return true;
        return false;
    }

    /** Tell the sorter what is live, and hand it the one slot's positions that moved. */
    pushSlots(base = null, chunkPos = null) {
        const flat = [];
        for (const [slot, cnt] of this.live) flat.push(slot * this.slotSize, cnt);
        const msg = { slots: flat, gen: this.generation };
        if (chunkPos) { msg.base = base; msg.chunk = chunkPos; }
        this.worker.postMessage(msg, chunkPos ? [chunkPos.buffer] : []);
    }

    /** Ask the worker to re-sort for this view. Cheap to call every frame. */
    requestSort(view) {
        if (this.sorting || !this.count) return;
        this.sorting = true;
        this.worker.postMessage({ view, gen: this.generation });
    }

    draw(view, proj, viewport, gain = 4.0, dilate = 0.6) {
        const gl = this.gl;
        if (!this.sortedCount) return 0;
        gl.useProgram(this.prog);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.tex);
        gl.uniform1i(this.u["u_splats"], 0);
        gl.uniformMatrix4fv(this.u["u_view"], false, view);
        gl.uniformMatrix4fv(this.u["u_proj"], false, proj);
        // focal in pixels, recovered from the projection matrix
        gl.uniform2f(this.u["u_focal"], proj[0] * viewport[0] * 0.5, proj[5] * viewport[1] * 0.5);
        gl.uniform2f(this.u["u_viewport"], viewport[0], viewport[1]);
        gl.uniform1f(this.u["u_gain"], gain);
        gl.uniform1f(this.u["u_fogDensity"], this.fogDensity);
        gl.uniform3f(this.u["u_fogColor"], this.fogColor[0], this.fogColor[1], this.fogColor[2]);
        gl.uniform1f(this.u["u_dilate"], dilate);

        gl.disable(gl.DEPTH_TEST);
        gl.enable(gl.BLEND);
        gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.bindVertexArray(this.vao);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.sortedCount);
        gl.bindVertexArray(null);
        gl.disable(gl.BLEND);
        gl.enable(gl.DEPTH_TEST);
        return this.sortedCount;
    }
}
