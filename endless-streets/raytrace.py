"""GPU ray tracer for the procedural town: closest hit against boxes and spheres (Triton), the ground
plane, procedural world-anchored textures, sun light with shadow rays, sky and haze.

Camera convention is OpenCV (columns of c2w rotation = right, down, forward), the same as the
reconstruction model and gsplat use, in a right-handed Y-up world.
"""
import math

import numpy as np
import torch
import triton
import triton.language as tl

import world as Wd
import cityassets as CA

TEXTURES = True        # use generated assets (facades, props, ground) where they exist
TEX_S = 640            # stored texture size (square; the surface restores the aspect)
TEX_LEVELS = 7         # 640 .. 10 px: ground seen at grazing angles needs the deep levels
ALPHA_S = 256          # cut-out alpha mask size used by the hit kernel
TEX_GAIN = 1.0         # photo albedo scale under the tracer's sun + sky light
PIX_ANGLE = 1.0 / 480  # radians per pixel of the walk / rig cameras (960 px over 90 deg)
_TEX_CACHE = {}


def _tex_mips(seed, key):
    """(mip chain [S,S,3] uint8 ..., alpha [ALPHA_S, ALPHA_S] uint8, aspect w/h of the photo), cached per process.
    Cut-outs: transparent pixels take the mean opaque colour, so filtering at the silhouette does not pull in
    the background."""
    ck = (seed, key)
    if ck not in _TEX_CACHE:
        from PIL import Image
        im = Image.open(CA.path(seed, key))
        aspect = im.width / im.height
        if im.mode == "RGBA":
            a = np.asarray(im).copy()
            op = a[..., 3] > 127
            if op.any():
                a[~op, :3] = a[op, :3].mean(0).astype(np.uint8)
            alpha = np.asarray(Image.fromarray(a[..., 3]).resize((ALPHA_S, ALPHA_S), Image.BILINEAR))
            im = Image.fromarray(a[..., :3])
        else:
            im = im.convert("RGB")
            alpha = np.full((ALPHA_S, ALPHA_S), 255, np.uint8)
        out, S = [], TEX_S
        for _ in range(TEX_LEVELS):
            out.append(np.asarray(im.resize((S, S), Image.BOX if S < TEX_S else Image.LANCZOS)))
            S //= 2
        if len(_TEX_CACHE) > 6000:
            _TEX_CACHE.pop(next(iter(_TEX_CACHE)))
        _TEX_CACHE[ck] = (out, alpha, aspect)
    return _TEX_CACHE[ck]


def _level(footprint, texel):
    return torch.log2((footprint / texel).clamp(min=1e-6)).floor().clamp(0, TEX_LEVELS - 1).long()


def _sample(scene, tid, u, v, lvl, wrap=False):
    """Bilinear lookup in the texture atlas (u, v in [0,1] of the photo; wrap: repeat)."""
    out = torch.zeros(u.shape[0], 3, device=u.device)
    for L in range(TEX_LEVELS):
        m = lvl == L
        if not m.any():
            continue
        tex = scene.tex[L]
        S = tex.shape[1]
        flat = tex.reshape(-1, 3)
        if wrap:
            x = torch.remainder(u[m], 1.0) * S - 0.5
            y = torch.remainder(v[m], 1.0) * S - 0.5
            x0, y0 = x.floor(), y.floor()
            fx, fy = (x - x0)[:, None], (y - y0)[:, None]
            x0, y0 = torch.remainder(x0, S).long(), torch.remainder(y0, S).long()
            x1, y1 = torch.remainder(x0 + 1, S), torch.remainder(y0 + 1, S)
        else:
            x = (u[m].clamp(0, 1) * S - 0.5).clamp(0, S - 1)
            y = (v[m].clamp(0, 1) * S - 0.5).clamp(0, S - 1)
            x0, y0 = x.floor().long(), y.floor().long()
            x1, y1 = (x0 + 1).clamp(max=S - 1), (y0 + 1).clamp(max=S - 1)
            fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        base = tid[m] * S * S

        def g(yy, xx):
            return flat[base + yy * S + xx].float()
        c = (g(y0, x0) * (1 - fx) + g(y0, x1) * fx) * (1 - fy) + (g(y1, x0) * (1 - fx) + g(y1, x1) * fx) * fy
        out[m] = c / 255.0
    # photos are display-referred; the renderer applies x^(1/1.15) at the end
    return out.pow(1.15)


def sample_box(scene, rows, q, nn, t):
    """Photo on the vertical faces of textured boxes (buildings, kiosks, poster columns, clocks): each face shows
    it left-to-right as seen from outside, v from the top of the box to its bottom."""
    b = scene.box[rows]
    xface = nn[:, 0].abs() > 0.5
    ux = (q[:, 2] - b[:, 2]) / (b[:, 5] - b[:, 2])
    uz = (q[:, 0] - b[:, 0]) / (b[:, 3] - b[:, 0])
    u = torch.where(xface, torch.where(nn[:, 0] > 0, 1 - ux, ux), torch.where(nn[:, 2] > 0, uz, 1 - uz))
    hgt = b[:, 4] - b[:, 1]
    v = (b[:, 4] - q[:, 1]) / hgt
    return _sample(scene, scene.tex_id[rows], u, v, _level(t * PIX_ANGLE, hgt / TEX_S))


SUN = np.array([0.42, 0.78, 0.46], np.float32)
SUN /= np.linalg.norm(SUN)
HORIZON = torch.tensor([0.80, 0.86, 0.92])
ZENITH = torch.tensor([0.36, 0.56, 0.84])
HAZE_M = 220.0
FAR = 400.0


@triton.jit
def _hit_kernel(o_ptr, d_ptr, box_ptr, sph_ptr, qd_ptr, al_ptr, t_out, id_out, R, NB, NS, NQ, tmin_eps,
                BR: tl.constexpr, BB: tl.constexpr, AS: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    m = r < R
    ox = tl.load(o_ptr + r * 3 + 0, mask=m, other=0.0)
    oy = tl.load(o_ptr + r * 3 + 1, mask=m, other=0.0)
    oz = tl.load(o_ptr + r * 3 + 2, mask=m, other=0.0)
    dx = tl.load(d_ptr + r * 3 + 0, mask=m, other=1.0)
    dy = tl.load(d_ptr + r * 3 + 1, mask=m, other=1.0)
    dz = tl.load(d_ptr + r * 3 + 2, mask=m, other=1.0)
    ix = 1.0 / tl.where(tl.abs(dx) < 1e-9, 1e-9, dx)
    iy = 1.0 / tl.where(tl.abs(dy) < 1e-9, 1e-9, dy)
    iz = 1.0 / tl.where(tl.abs(dz) < 1e-9, 1e-9, dz)
    best = tl.full((BR,), 1e30, tl.float32)
    bid = tl.full((BR,), -1, tl.int32)
    for b0 in range(0, NB, BB):
        b = b0 + tl.arange(0, BB)
        mb = b < NB
        x0 = tl.load(box_ptr + b * 7 + 0, mask=mb, other=0.0)
        y0 = tl.load(box_ptr + b * 7 + 1, mask=mb, other=0.0)
        z0 = tl.load(box_ptr + b * 7 + 2, mask=mb, other=0.0)
        x1 = tl.load(box_ptr + b * 7 + 3, mask=mb, other=-1.0)
        y1 = tl.load(box_ptr + b * 7 + 4, mask=mb, other=-1.0)
        z1 = tl.load(box_ptr + b * 7 + 5, mask=mb, other=-1.0)
        tx0 = (x0[None, :] - ox[:, None]) * ix[:, None]
        tx1 = (x1[None, :] - ox[:, None]) * ix[:, None]
        ty0 = (y0[None, :] - oy[:, None]) * iy[:, None]
        ty1 = (y1[None, :] - oy[:, None]) * iy[:, None]
        tz0 = (z0[None, :] - oz[:, None]) * iz[:, None]
        tz1 = (z1[None, :] - oz[:, None]) * iz[:, None]
        tn = tl.maximum(tl.maximum(tl.minimum(tx0, tx1), tl.minimum(ty0, ty1)), tl.minimum(tz0, tz1))
        tf = tl.minimum(tl.minimum(tl.maximum(tx0, tx1), tl.maximum(ty0, ty1)), tl.maximum(tz0, tz1))
        ok = (tf >= tn) & (tf > tmin_eps) & mb[None, :]
        th = tl.where(tn > tmin_eps, tn, tf)
        th = tl.where(ok, th, 1e30)
        cmin = tl.min(th, axis=1)
        carg = tl.argmin(th, axis=1)
        upd = cmin < best
        best = tl.where(upd, cmin, best)
        bid = tl.where(upd, b0 + carg.to(tl.int32), bid)
    for s0 in range(0, NS, BB):
        s = s0 + tl.arange(0, BB)
        ms = s < NS
        cx = tl.load(sph_ptr + s * 5 + 0, mask=ms, other=0.0)
        cy = tl.load(sph_ptr + s * 5 + 1, mask=ms, other=0.0)
        cz = tl.load(sph_ptr + s * 5 + 2, mask=ms, other=0.0)
        rr = tl.load(sph_ptr + s * 5 + 3, mask=ms, other=0.0)
        lx = ox[:, None] - cx[None, :]
        ly = oy[:, None] - cy[None, :]
        lz = oz[:, None] - cz[None, :]
        bb = lx * dx[:, None] + ly * dy[:, None] + lz * dz[:, None]
        cc = lx * lx + ly * ly + lz * lz - (rr * rr)[None, :]
        disc = bb * bb - cc
        sq = tl.sqrt(tl.maximum(disc, 0.0))
        t0 = -bb - sq
        t1 = -bb + sq
        th = tl.where(t0 > tmin_eps, t0, t1)
        ok = (disc >= 0) & (th > tmin_eps) & ms[None, :]
        th = tl.where(ok, th, 1e30)
        cmin = tl.min(th, axis=1)
        carg = tl.argmin(th, axis=1)
        upd = cmin < best
        best = tl.where(upd, cmin, best)
        bid = tl.where(upd, NB + s0 + carg.to(tl.int32), bid)
    # vertical billboards (generated cut-outs): plane through (cx, cz) along (ex, ez), alpha-tested
    for q0 in range(0, NQ, BB):
        q = q0 + tl.arange(0, BB)
        mq = q < NQ
        qx = tl.load(qd_ptr + q * 8 + 0, mask=mq, other=0.0)
        qy0 = tl.load(qd_ptr + q * 8 + 1, mask=mq, other=0.0)
        qz = tl.load(qd_ptr + q * 8 + 2, mask=mq, other=0.0)
        ex = tl.load(qd_ptr + q * 8 + 3, mask=mq, other=1.0)
        ez = tl.load(qd_ptr + q * 8 + 4, mask=mq, other=0.0)
        hw = tl.load(qd_ptr + q * 8 + 5, mask=mq, other=0.0)
        hh = tl.load(qd_ptr + q * 8 + 6, mask=mq, other=0.0)
        qt = tl.load(qd_ptr + q * 8 + 7, mask=mq, other=0.0).to(tl.int32)
        den = dx[:, None] * (-ez[None, :]) + dz[:, None] * ex[None, :]
        num = (qx[None, :] - ox[:, None]) * (-ez[None, :]) + (qz[None, :] - oz[:, None]) * ex[None, :]
        th = num / tl.where(tl.abs(den) < 1e-9, 1e-9, den)
        px = ox[:, None] + th * dx[:, None]
        py = oy[:, None] + th * dy[:, None]
        pz = oz[:, None] + th * dz[:, None]
        u = (px - qx[None, :]) * ex[None, :] + (pz - qz[None, :]) * ez[None, :]
        hv = py - qy0[None, :]
        ok = (th > tmin_eps) & (tl.abs(u) <= hw[None, :]) & (hv >= 0) & (hv <= hh[None, :]) & mq[None, :] & (tl.abs(den) >= 1e-9)
        iu = ((u / tl.maximum(hw[None, :], 1e-6) * 0.5 + 0.5) * AS).to(tl.int32)
        iu = tl.minimum(tl.maximum(iu, 0), AS - 1)
        iv = ((1.0 - hv / tl.maximum(hh[None, :], 1e-6)) * AS).to(tl.int32)
        iv = tl.minimum(tl.maximum(iv, 0), AS - 1)
        al = tl.load(al_ptr + qt[None, :] * (AS * AS) + iv * AS + iu, mask=ok, other=0)
        ok = ok & (al > 127)
        th = tl.where(ok, th, 1e30)
        cmin = tl.min(th, axis=1)
        carg = tl.argmin(th, axis=1)
        upd = cmin < best
        best = tl.where(upd, cmin, best)
        bid = tl.where(upd, NB + NS + q0 + carg.to(tl.int32), bid)
    tl.store(t_out + r, best, mask=m)
    tl.store(id_out + r, bid, mask=m)


class Scene:
    """Primitives of a window of the town, resident on one device, with the generated assets that exist so far:
    photos on boxes (buildings, kiosks, posters, clocks), cut-out billboards (trees, statues, benches) that
    replace their stand-in primitives, and ground materials per street segment / block."""
    def __init__(self, xmin, xmax, zmin, zmax, seed=0, device="cuda"):
        box, boxp, sph, sphp, boxk, sphk, props = Wd.window(xmin, xmax, zmin, zmax, seed)
        self.device = device
        self.box = torch.from_numpy(box).to(device).contiguous()
        self.boxp = torch.from_numpy(boxp).to(device)
        self.sph = torch.from_numpy(sph).to(device).contiguous()
        self.sphp = torch.from_numpy(sphp).to(device)
        self.nb, self.ns = self.box.shape[0], self.sph.shape[0]
        self.seed = seed
        self.boxk, self.sphk, self.props = boxk, sphk, props
        self.tex_id = torch.full((self.nb,), -1, dtype=torch.long, device=device)
        self.tex = None                     # list of mip levels [T, S, S, 3] uint8
        self.alpha = torch.full((1,), 255, dtype=torch.uint8, device=device)
        self.tex_keys = []
        self.quads = torch.zeros(0, 8, device=device)
        self.nq = 0
        self.gi0, self.gj0 = math.floor(xmin / Wd.P) - 1, math.floor(zmin / Wd.P) - 1
        ni, nj = math.floor(xmax / Wd.P) + 2 - self.gi0, math.floor(zmax / Wd.P) + 2 - self.gj0
        self.road_tid = torch.full((ni, nj, 2), -1, dtype=torch.long, device=device)
        self.walk_tid = torch.full((ni, nj), -1, dtype=torch.long, device=device)
        self.uses = {}
        if TEXTURES:
            add = lambda a, u: self.uses.setdefault(a["key"], []).append(u)
            for i in np.nonzero(box[:, 6] == Wd.M_FACADE)[0]:
                add(CA.facade(box[i], boxp[i], seed), ("box", int(i)))
            for pi, pr in enumerate(props):
                a = CA.prop_asset(pr, seed)
                if a is None:
                    continue
                if int(pr[0]) in Wd.BILLBOARD_PROPS:
                    add(a, ("bill", pi))
                else:
                    for r in np.nonzero(boxk == pi)[0]:
                        add(a, ("box", int(r)))
            for ii in range(ni):
                for jj in range(nj):
                    i, j = ii + self.gi0, jj + self.gj0
                    add(CA.road_asset(i, j, 0, seed), ("road", ii, jj, 0))
                    add(CA.road_asset(i, j, 1, seed), ("road", ii, jj, 1))
                    add(CA.walk_asset(i, j, seed), ("walk", ii, jj))
        self.refresh_textures()

    def refresh_textures(self):
        """Load generated assets that exist on disk and are not loaded yet. Returns how many were added."""
        new = [k for k in self.uses if k not in self._loaded() and CA.have(self.seed, k)]
        if not new:
            return 0
        levels = [[] for _ in range(TEX_LEVELS)]
        alphas, aspects = [], []
        for k in new:
            mips, al, asp = _tex_mips(self.seed, k)
            for L, a in enumerate(mips):
                levels[L].append(torch.from_numpy(a.copy()))
            alphas.append(torch.from_numpy(al.copy()).reshape(-1))
            aspects.append(asp)
        add = [torch.stack(l).to(self.device) for l in levels]
        base = len(self.tex_keys)
        self.tex = add if self.tex is None else [torch.cat([a, b]) for a, b in zip(self.tex, add)]
        al = torch.cat(alphas).to(self.device)
        self.alpha = al if base == 0 else torch.cat([self.alpha, al])
        self.tex_keys += new
        quads, hide_b, hide_s = [], [], []
        for n, k in enumerate(new):
            t = base + n
            for u in self.uses[k]:
                if u[0] == "box":
                    self.tex_id[u[1]] = t
                elif u[0] == "road":
                    self.road_tid[u[1], u[2], u[3]] = t
                elif u[0] == "walk":
                    self.walk_tid[u[1], u[2]] = t
                else:
                    pi = u[1]
                    pr = self.props[pi]
                    y0, h, ang = CA.prop_shape(pr, self.seed)
                    if h is None:                       # bench: width fixed by its box, height from the photo
                        hw = float(pr[5]); h = 2 * hw / aspects[n]
                    else:
                        hw = h * aspects[n] / 2
                    angs = (ang, ang + math.pi / 2) if pr[7] > 0 else (ang,)
                    for a in angs:
                        quads.append([float(pr[1]), y0, float(pr[3]), math.cos(a), math.sin(a), hw, h, float(t)])
                    hide_b += list(np.nonzero(self.boxk == pi)[0])
                    hide_s += list(np.nonzero(self.sphk == pi)[0])
        if quads:
            self.quads = torch.cat([self.quads, torch.tensor(quads, device=self.device)]).contiguous()
            self.nq = self.quads.shape[0]
        if hide_b:          # stand-in primitives of props now drawn as cut-outs: moved far below the ground
            rb = torch.tensor(hide_b, device=self.device)
            self.box[rb, 1] = -1e4
            self.box[rb, 4] = -1e4 + 1e-3
        if hide_s:
            self.sph[torch.tensor(hide_s, device=self.device), 1] = -1e4
        return len(new)

    def _loaded(self):
        if getattr(self, "_lset_n", -1) != len(self.tex_keys):
            self._lset, self._lset_n = set(self.tex_keys), len(self.tex_keys)
        return self._lset

    def missing_textures(self):
        return [k for k in self.uses if k not in self._loaded()]

    def intersect(self, o, d, eps=1e-3):
        """Closest hit: t [R] (inf if none), id [R] (-1 none, < nb box, < nb + ns sphere, else billboard).
        Includes the ground plane (id -2)."""
        R = o.shape[0]
        t = torch.empty(R, device=o.device, dtype=torch.float32)
        idx = torch.empty(R, device=o.device, dtype=torch.int32)
        box = self.box if self.nb else torch.zeros(1, 7, device=o.device)
        sph = self.sph if self.ns else torch.zeros(1, 5, device=o.device)
        qd = self.quads if self.nq else torch.zeros(1, 8, device=o.device)
        grid = (triton.cdiv(R, 128),)
        _hit_kernel[grid](o.contiguous(), d.contiguous(), box, sph, qd, self.alpha, t, idx, R, self.nb, self.ns, self.nq,
                          eps, BR=128, BB=32, AS=ALPHA_S)
        tg = torch.where(d[:, 1] < -1e-6, -o[:, 1] / d[:, 1].clamp(max=-1e-6), torch.full_like(t, 1e30))
        tg = torch.where(tg > eps, tg, torch.full_like(t, 1e30))
        ground = tg < t
        t = torch.where(ground, tg, t)
        idx = torch.where(ground, torch.full_like(idx, -2), idx)
        t = torch.where(t > FAR, torch.full_like(t, float("inf")), t)
        idx = torch.where(torch.isinf(t), torch.full_like(idx, -1), idx)
        return t, idx.long()

    def surface_distance(self, pts):
        """Distance from points to the nearest surface (ground, boxes, spheres)."""
        best = pts[:, 1].abs()
        for k in range(0, self.nb, 256):
            b = self.box[k:k + 256]
            lo, hi = b[None, :, 0:3], b[None, :, 3:6]
            p = pts[:, None, :]
            q = torch.maximum(lo - p, p - hi)
            out = q.clamp(min=0).norm(dim=-1)
            inside = q.max(dim=-1).values.clamp(max=0).abs()
            best = torch.minimum(best, torch.where(out > 0, out, inside).min(dim=1).values)
        for k in range(0, self.nq, 256):
            qd = self.quads[k:k + 256]
            dp = pts[:, None, :] - qd[None, :, [0, 1, 2]]
            along = dp[..., 0] * qd[None, :, 3] + dp[..., 2] * qd[None, :, 4]
            perp = -dp[..., 0] * qd[None, :, 4] + dp[..., 2] * qd[None, :, 3]
            ex = (along.abs() - qd[None, :, 5]).clamp(min=0)
            ey = torch.maximum(-dp[..., 1], dp[..., 1] - qd[None, :, 6]).clamp(min=0)
            best = torch.minimum(best, (perp * perp + ex * ex + ey * ey).sqrt().min(dim=1).values)
        for k in range(0, self.ns, 256):
            s = self.sph[k:k + 256]
            dd = ((pts[:, None, :] - s[None, :, :3]).norm(dim=-1) - s[None, :, 3]).abs()
            best = torch.minimum(best, dd.min(dim=1).values)
        return best


# ------------------------------------------------------------------ procedural textures
def _hash3(a, b, c):
    """Float hash in [0,1) of integer-valued tensors (exact for |values| < 2^20)."""
    x = (a.long() * 73856093) ^ (b.long() * 19349663) ^ (c.long() * 83492791)
    x = (x ^ (x >> 13)) * 1274126177
    x = x ^ (x >> 16)
    return (x & 0xFFFFFF).float() / 16777216.0


def _vnoise(x, y, s):
    xi, yi = torch.floor(x), torch.floor(y)
    xf, yf = x - xi, y - yi
    u, v = xf * xf * (3 - 2 * xf), yf * yf * (3 - 2 * yf)
    a = _hash3(xi, yi, s); b = _hash3(xi + 1, yi, s); c = _hash3(xi, yi + 1, s); d = _hash3(xi + 1, yi + 1, s)
    return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v


def _fbm(x, y, s, oct=4):
    acc, amp, f = 0.0, 0.5, 1.0
    for k in range(oct):
        acc = acc + amp * _vnoise(x * f, y * f, s + k * 17)
        amp, f = amp * 0.5, f * 2.03
    return acc


def sky(d):
    up = d[:, 1].clamp(-1, 1)
    t = up.clamp(min=0).pow(0.6)[:, None]
    col = HORIZON.to(d.device) * (1 - t) + ZENITH.to(d.device) * t
    k = (60.0 / up.clamp(min=0.04))[:, None]
    cx, cz = d[:, 0:1] * k, d[:, 2:3] * k
    cl = _fbm(cx[:, 0] * 0.012, cz[:, 0] * 0.012, torch.full_like(up, 5), 5)
    cl = ((cl - 0.48) * 3.0).clamp(0, 1) * (up > 0).float() * (1 - (-up * 8).exp().clamp(max=1) * 0)
    col = col * (1 - 0.8 * cl[:, None]) + torch.tensor([0.97, 0.97, 0.98], device=d.device) * 0.8 * cl[:, None]
    sun = torch.tensor(SUN, device=d.device)
    glow = (d @ sun).clamp(min=0).pow(200)[:, None]
    below = (up < 0).float()[:, None]
    col = col * (1 - below) + HORIZON.to(d.device) * 0.9 * below
    return (col + glow * 0.6).clamp(0, 1)


def _glyph(u, v, sd):
    """Letter-like strokes: a 3x5 cell grid per character, cells on/off by hash."""
    ci = torch.floor(u / 0.42)
    lu = (u / 0.42 - ci) * 4.0          # 0..4 : 3 columns + gap
    lv = v * 5.0
    col, row = torch.floor(lu), torch.floor(lv)
    on = (_hash3(ci, col * 7 + row, sd) > 0.45) & (col < 3) & (row >= 0) & (row < 5)
    space = _hash3(ci, torch.full_like(ci, 991), sd) < 0.15
    return on & ~space


def shade_facade(p, n, prm, vd):
    """Building faces: shop fronts, sign band, windows, wall material. World-anchored."""
    wall = prm[:, 0:3]
    ws, code, bay, sd, height = prm[:, 3], prm[:, 4], prm[:, 5], prm[:, 6], prm[:, 7]
    style = torch.remainder(code, 4)
    signc = torch.tensor(Wd.PALETTE_SIGN, device=p.device)[torch.floor(code / 4).long().clamp(0, len(Wd.PALETTE_SIGN) - 1)]
    side = n[:, 0].abs() > 0.5
    u = torch.where(side, p[:, 2], p[:, 0])
    v = p[:, 1]
    # wall material
    nz = _fbm(u * 1.7, v * 1.7, sd)
    col = wall * (0.9 + 0.2 * nz[:, None])
    brick = style == 1
    bu, bv = u / 0.25, v / 0.075
    row = torch.floor(bv)
    mortar = ((bv - row) < 0.14) | ((torch.remainder(bu + 0.5 * torch.remainder(row, 2), 1.0)) < 0.06)
    bvar = 0.85 + 0.25 * _hash3(torch.floor(bu + 0.5 * torch.remainder(row, 2)), row, sd)
    col = torch.where((brick & ~mortar)[:, None], col * bvar[:, None], col)
    col = torch.where((brick & mortar)[:, None], torch.full_like(col, 0.72), col)
    panel = style == 2
    pl = (torch.remainder(u, 1.5) < 0.03) | (torch.remainder(v, 1.6) < 0.03)
    col = torch.where((panel & pl)[:, None], col * 0.7, col)
    # upper floors: windows
    upper = v >= Wd.GF
    fl = torch.floor((v - Wd.GF) / Wd.FH)
    ly = (v - Wd.GF) - fl * Wd.FH
    wu = torch.remainder(u, ws) / ws
    wi = torch.floor(u / ws)
    inwin = upper & (ly > 0.85) & (ly < 2.45) & (wu > 0.24) & (wu < 0.76) & (v < height - 0.6)
    frame = inwin & ((ly < 0.95) | (ly > 2.35) | (wu < 0.29) | (wu > 0.71) | ((wu - 0.5).abs() < 0.02))
    sill = upper & (ly > 0.72) & (ly < 0.85) & (wu > 0.21) & (wu < 0.79) & (v < height - 0.6)
    hv = _hash3(wi, fl, sd)
    fres = 0.35 * (1 - (vd * n).sum(-1).abs()).clamp(0, 1).pow(3)[:, None]
    glass = torch.tensor([0.16, 0.22, 0.27], device=p.device) * (0.7 + 0.6 * hv[:, None])
    curtain = (hv > 0.72)[:, None] & (ly > 1.5)[:, None]
    glass = torch.where(curtain, torch.tensor([0.78, 0.74, 0.66], device=p.device) * (0.6 + 0.4 * hv[:, None]), glass)
    glass = glass * (1 - fres) + sky(torch.nn.functional.normalize(vd - 2 * (vd * n).sum(-1, keepdim=True) * n, dim=-1)) * fres
    col = torch.where(inwin[:, None], glass, col)
    col = torch.where(frame[:, None], torch.full_like(col, 0.88), col)
    col = torch.where(sill[:, None], col * 0.75, col)
    # cornice
    corn = (v > height - 0.45) & (v < height - 0.2)
    col = torch.where(corn[:, None], col * 0.72, col)
    # ground floor: shop fronts between pilasters, sign band
    gnd = v < Wd.GF
    bu = torch.remainder(u, bay) / bay
    bi = torch.floor(u / bay)
    pil = gnd & ((bu < 0.07) | (bu > 0.93))
    door = _hash3(bi, torch.zeros_like(bi), sd) < 0.25
    shop = gnd & ~pil & (v > 0.35) & (v < 3.0)
    gshop = torch.tensor([0.10, 0.13, 0.15], device=p.device) * (0.8 + 0.5 * _fbm(u * 0.8, v * 3, sd + 3)[:, None])
    disp = (_hash3(bi, torch.floor(v * 2), sd + 1) > 0.6) & (v < 1.3)
    gshop = torch.where(disp[:, None], torch.tensor(Wd.PALETTE_BRIGHT, device=p.device)[(hv * 6).long().clamp(0, 5)] * 0.55, gshop)
    gshop = gshop * (1 - fres) + sky(torch.nn.functional.normalize(vd - 2 * (vd * n).sum(-1, keepdim=True) * n, dim=-1)) * fres
    mullion = shop & ((torch.remainder(u, bay / 2) / (bay / 2) < 0.03) | ((v > 2.2) & (v < 2.27)))
    col = torch.where(shop[:, None], gshop, col)
    col = torch.where((shop & door & (bu > 0.35) & (bu < 0.65))[:, None], torch.tensor([0.30, 0.20, 0.14], device=p.device).expand_as(col), col)
    col = torch.where(mullion[:, None], torch.full_like(col, 0.25), col)
    plinth = gnd & (v <= 0.35)
    col = torch.where(plinth[:, None], col * 0.55, col)
    band = (v > 3.2) & (v < 4.05)
    sv = (v - 3.32) / 0.6
    letters = band & (sv > 0) & (sv < 1) & _glyph(u + sd * 0.37, sv, sd)
    col = torch.where(band[:, None], signc, col)
    lum = (signc * torch.tensor([0.3, 0.59, 0.11], device=p.device)).sum(-1, keepdim=True)
    lc = torch.where(lum > 0.5, torch.full_like(col, 0.08), torch.full_like(col, 0.95))
    col = torch.where(letters[:, None], lc, col)
    awning = (v > 2.95) & (v < 3.2) & ~pil
    col = torch.where(awning[:, None], signc * 0.6, col)
    return col


def ground_photo(scene, x, z, road_x, road_z, walk, t, d):
    """Generated material under each ground point (asphalt per street segment, paving per block), or -1 where
    none is loaded yet. -> (colour [R,3], has [R])."""
    P = Wd.P
    ni, nj = scene.walk_tid.shape
    ix, jz = torch.floor(x / P).long() - scene.gi0, torch.floor(z / P).long() - scene.gj0
    ir, jr = torch.round(x / P).long() - scene.gi0, torch.round(z / P).long() - scene.gj0

    def look(tab, a, b, *c):
        ok = (a >= 0) & (a < ni) & (b >= 0) & (b < nj)
        v = tab[a.clamp(0, ni - 1), b.clamp(0, nj - 1), *c] if c else tab[a.clamp(0, ni - 1), b.clamp(0, nj - 1)]
        return torch.where(ok, v, torch.full_like(v, -1))
    tid = torch.full_like(ix, -1)
    tid = torch.where(walk, look(scene.walk_tid, ix, jz), tid)
    tid = torch.where(road_z, look(scene.road_tid, ir, jz, 1), tid)
    tid = torch.where(road_x, look(scene.road_tid, ix, jr, 0), tid)        # crossings: the x street's asphalt
    has = tid >= 0
    col = torch.zeros(x.shape[0], 3, device=x.device)
    if has.any():
        road = (road_x | road_z)[has]
        per = torch.where(road, torch.full_like(x[has], CA.ROAD_PERIOD), torch.full_like(x[has], CA.PAVE_PERIOD))
        # no anisotropic filtering: dividing by sqrt(cos) instead of cos keeps grazing ground sharp, a little aliased
        foot = t[has] * PIX_ANGLE / d[has][:, 1].abs().clamp(min=0.03).sqrt()
        lvl = _level(foot, per / TEX_S)
        xh, zh = x[has], z[has]
        c1 = _sample(scene, tid[has], xh / per, zh / per, lvl, wrap=True)
        # asphalt: a second, rotated and rescaled layer blended by low-frequency noise hides the repeat
        c2 = _sample(scene, tid[has], (zh * 0.77 + 0.31) / per, (-xh * 0.77 + 0.57) / per, lvl, wrap=True)
        w = _fbm(xh * 0.12, zh * 0.12, torch.full_like(xh, 5))[:, None].clamp(0, 1)
        col[has] = torch.where(road[:, None], c1 * (1 - w) + c2 * w, c1) * TEX_GAIN
    return col, has


def shade_ground(p, scene=None, t=None, d=None):
    x, z = p[:, 0], p[:, 2]
    lx = torch.remainder(x + Wd.P / 2, Wd.P) - Wd.P / 2
    lz = torch.remainder(z + Wd.P / 2, Wd.P) - Wd.P / 2
    ax, az = lx.abs(), lz.abs()
    road_x = az < Wd.ROAD          # street running along x
    road_z = ax < Wd.ROAD
    road = road_x | road_z
    walk = ~road & ((az < Wd.HW) | (ax < Wd.HW))
    n = _fbm(x * 2.0, z * 2.0, torch.full_like(x, 11))
    asph = torch.stack([0.24 + 0.08 * n, 0.24 + 0.08 * n, 0.26 + 0.08 * n], -1)
    # sidewalk paving
    tu, tv = torch.remainder(x, 0.6) / 0.6, torch.remainder(z, 0.6) / 0.6
    joint = (tu < 0.05) | (tv < 0.05)
    tile = 0.62 + 0.12 * _hash3(torch.floor(x / 0.6), torch.floor(z / 0.6), torch.full_like(x, 3))
    pave = torch.stack([tile, tile * 0.97, tile * 0.92], -1)
    pave = torch.where(joint[:, None], pave * 0.75, pave)
    curb = ~road & (((az > Wd.ROAD) & (az < Wd.ROAD + 0.22) & (ax > Wd.HW)) | ((ax > Wd.ROAD) & (ax < Wd.ROAD + 0.22) & (az > Wd.HW)))
    pave = torch.where(curb[:, None], torch.full_like(pave, 0.80), pave)
    col = torch.where(road[:, None], asph, pave)
    # block interiors (courtyards) never seen, but keep them plain
    inner = ~road & ~walk
    col = torch.where(inner[:, None], torch.full_like(col, 0.4), col)
    if scene is not None and scene.tex is not None:
        gc, has = ground_photo(scene, x, z, road_x, road_z, walk & ~curb, t, d)
        col = torch.where(has[:, None], gc, col)
    white = torch.full_like(col, 0.88)
    cross = road_x & road_z
    # centre dashes
    dash_x = road_x & ~road_z & (az < 0.08) & (torch.remainder(x, 6.0) < 3.0) & (ax > Wd.HW + 4)
    dash_z = road_z & ~road_x & (ax < 0.08) & (torch.remainder(z, 6.0) < 3.0) & (az > Wd.HW + 4)
    # zebra crossings on every arm, just outside the junction box
    zeb_x = road_z & (az > Wd.HW + 0.3) & (az < Wd.HW + 3.3) & (torch.remainder(x + 0.25, 1.0) < 0.5)
    zeb_z = road_x & (ax > Wd.HW + 0.3) & (ax < Wd.HW + 3.3) & (torch.remainder(z + 0.25, 1.0) < 0.5)
    stop_x = road_z & ~cross & (az > Wd.HW + 3.6) & (az < Wd.HW + 3.9)
    stop_z = road_x & ~cross & (ax > Wd.HW + 3.6) & (ax < Wd.HW + 3.9)
    mark = dash_x | dash_z | zeb_x | zeb_z | stop_x | stop_z
    col = torch.where(mark[:, None], white * (0.9 + 0.1 * n[:, None]), col)
    return col


def shade_misc(mat, p, n, prm):
    base = prm[:, 0:3]
    col = base.clone()
    nz = _fbm(p[:, 0] * 3 + p[:, 2] * 3, p[:, 1] * 3, prm[:, 4] if prm.shape[1] > 4 else torch.zeros_like(p[:, 0]))
    crown = mat == Wd.M_CROWN
    leaf = _fbm(p[:, 0] * 4.0 + p[:, 1] * 2, p[:, 2] * 4.0 - p[:, 1] * 2, prm[:, 3])
    col = torch.where(crown[:, None], base * (0.55 + 0.9 * leaf[:, None]), col)
    trunk = mat == Wd.M_TRUNK
    col = torch.where(trunk[:, None], base * (0.8 + 0.4 * _fbm(p[:, 1] * 6, p[:, 0] * 6 + p[:, 2] * 6, torch.zeros_like(p[:, 0]))[:, None]), col)
    stone = mat == Wd.M_STONE
    col = torch.where(stone[:, None], base * (0.85 + 0.3 * nz[:, None]), col)
    # kiosk: coloured body with a window strip, clock face
    kiosk = mat == Wd.M_KIOSK
    side = n[:, 0].abs() > 0.5
    u = torch.where(side, p[:, 2], p[:, 0])
    win = kiosk & (prm[:, 3] == 0) & (p[:, 1] > 1.0) & (p[:, 1] < 1.9) & (n[:, 1].abs() < 0.5) & (torch.remainder(u, 0.9) < 0.75)
    col = torch.where(win[:, None], torch.full_like(col, 0.15), col)
    clock = kiosk & (prm[:, 3] == 2) & (n[:, 1].abs() < 0.5)
    cu = torch.remainder(u + 0.45, 0.9) - 0.45
    cv = p[:, 1] - 4.05
    r = (cu * cu + cv * cv).sqrt()
    face = clock & (r < 0.36)
    ang = torch.atan2(cv, cu)
    hand = face & (((ang - 1.2).abs() < 0.08) & (r < 0.22) | ((ang + 0.3).abs() < 0.06) & (r < 0.32))
    ring = face & (r > 0.32)
    col = torch.where(face[:, None], torch.full_like(col, 0.95), col)
    col = torch.where((hand | ring)[:, None], torch.full_like(col, 0.08), col)
    # advertising column: poster bands
    ad = mat == Wd.M_ADCOL
    band = torch.floor(p[:, 1] / 0.8)
    pc = torch.tensor(Wd.PALETTE_BRIGHT, device=p.device)[(_hash3(band, torch.floor(u / 0.55), prm[:, 3]) * 6).long().clamp(0, 5)]
    stripe = torch.remainder(p[:, 1], 0.8) < 0.06
    col = torch.where(ad[:, None], torch.where(stripe[:, None], torch.full_like(col, 0.9), pc * (0.7 + 0.3 * nz[:, None])), col)
    lamp = mat == Wd.M_LAMP
    col = torch.where(lamp[:, None], base, col)
    bronze = mat == Wd.M_BRONZE
    col = torch.where(bronze[:, None], base * (0.7 + 0.6 * nz[:, None]), col)
    return col


class Raytracer:
    def __init__(self, scene):
        self.s = scene

    @torch.no_grad()
    def trace(self, o, d, shadows=True, want_depth=False, haze=True):
        """o, d: [R,3] world rays (d unit). -> colour [R,3] linear-ish sRGB in [0,1], t [R], hit id."""
        s = self.s
        t, idx = s.intersect(o, d)
        hit = idx != -1
        p = o + d * torch.where(hit, t, torch.zeros_like(t))[:, None]
        n = torch.zeros_like(p)
        col = torch.zeros_like(p)
        isbox = (idx >= 0) & (idx < s.nb)
        issph = (idx >= s.nb) & (idx < s.nb + s.ns)
        isq = idx >= s.nb + s.ns
        isgnd = idx == -2
        n[isgnd] = torch.tensor([0.0, 1.0, 0.0], device=p.device)
        if isbox.any():
            b = s.box[idx[isbox]]
            q = p[isbox]
            dl = torch.stack([(q[:, 0] - b[:, 0]).abs(), (q[:, 0] - b[:, 3]).abs(), (q[:, 1] - b[:, 1]).abs(),
                              (q[:, 1] - b[:, 4]).abs(), (q[:, 2] - b[:, 2]).abs(), (q[:, 2] - b[:, 5]).abs()], -1)
            f = dl.argmin(-1)
            nn = torch.zeros_like(q)
            sign = torch.where(f % 2 == 0, -1.0, 1.0)
            nn[torch.arange(q.shape[0], device=q.device), f // 2] = sign
            n[isbox] = nn
            mat = b[:, 6].long()
            prm = s.boxp[idx[isbox]]
            c = shade_misc(mat, q, nn, prm)
            fac = mat == Wd.M_FACADE
            if fac.any():
                cf = shade_facade(q[fac], nn[fac], prm[fac], d[isbox][fac])
                roof = nn[fac][:, 1] > 0.5
                cf = torch.where(roof[:, None], torch.full_like(cf, 0.42), cf)
                c[fac] = cf
            if s.tex is not None:
                rows = idx[isbox]
                tm = (s.tex_id[rows] >= 0) & (nn[:, 1].abs() < 0.5)
                if tm.any():
                    c[tm] = sample_box(s, rows[tm], q[tm], nn[tm], t[isbox][tm]) * TEX_GAIN
            col[isbox] = c
        if issph.any():
            sp = s.sph[idx[issph] - s.nb]
            q = p[issph]
            nn = torch.nn.functional.normalize(q - sp[:, :3], dim=-1)
            n[issph] = nn
            col[issph] = shade_misc(sp[:, 4].long(), q, nn, s.sphp[idx[issph] - s.nb])
        if isq.any():
            qd = s.quads[idx[isq] - s.nb - s.ns]
            q = p[isq]
            nrm = torch.stack([-qd[:, 4], torch.zeros_like(qd[:, 4]), qd[:, 3]], -1)
            nn = nrm * -torch.sign((d[isq] * nrm).sum(-1, keepdim=True) + 1e-9)     # the side facing the ray
            n[isq] = nn
            u = ((q[:, 0] - qd[:, 0]) * qd[:, 3] + (q[:, 2] - qd[:, 2]) * qd[:, 4]) / qd[:, 5] * 0.5 + 0.5
            v = 1 - (q[:, 1] - qd[:, 1]) / qd[:, 6]
            col[isq] = _sample(s, qd[:, 7].long(), u, v, _level(t[isq] * PIX_ANGLE, qd[:, 6] / TEX_S)) * TEX_GAIN
        if isgnd.any():
            col[isgnd] = shade_ground(p[isgnd], s, t[isgnd], d[isgnd])
        # lighting
        sun = torch.tensor(SUN, device=p.device)
        ndl = (n * sun).sum(-1).clamp(min=0)
        lit = torch.ones_like(ndl)
        if shadows and hit.any():
            hi = torch.nonzero(hit & (ndl > 0)).squeeze(1)
            if hi.numel():
                so = p[hi] + n[hi] * 0.02
                ts, _ = s.intersect(so, sun.expand(hi.numel(), 3).contiguous(), eps=1e-3)
                lit[hi] = torch.isinf(ts).float()
        amb = 0.40 + 0.18 * n[:, 1].clamp(min=-1)
        lamp = (idx >= 0) & (idx < s.nb)
        light = (amb + 0.95 * ndl * lit)[:, None]
        c = col * light
        if lamp.any():
            em = s.box[idx[lamp]][:, 6] == Wd.M_LAMP
            cl = c[lamp]
            cl[em] = col[lamp][em]
            c[lamp] = cl
        if haze:
            hz = (1 - torch.exp(-torch.where(hit, t, torch.zeros_like(t)) / HAZE_M))[:, None]
            c = c * (1 - hz) + HORIZON.to(p.device) * hz
        c = torch.where(hit[:, None], c, sky(d))
        self.last_normal = n
        return c.clamp(0, 1), torch.where(hit, t, torch.full_like(t, float("inf"))), idx


def camera_basis(yaw, pitch):
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    f = np.array([sy * cp, sp, cy * cp], dtype=np.float64)
    r = np.array([-cy, 0.0, sy], dtype=np.float64)
    d = np.cross(f, r)
    return np.stack([r, d, f], axis=1).astype(np.float32)


def c2w(yaw, pitch, pos):
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = camera_basis(yaw, pitch)
    m[:3, 3] = np.asarray(pos, np.float32)
    return m


def K_of(W, H, fov_x):
    fx = (W / 2) / math.tan(fov_x / 2)
    return np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], np.float32)


@torch.no_grad()
def render(tracer, c2w_m, K, W, H, shadows=True, chunk=1 << 19, normals=False, haze=True, ss=1):
    """Image [H,W,3] in [0,1] and z-depth [H,W] in metres (0 for sky) at an OpenCV c2w."""
    dev = tracer.s.device
    c2w_t = torch.as_tensor(c2w_m, device=dev, dtype=torch.float32)
    Kt = torch.as_tensor(K, device=dev, dtype=torch.float32)
    # ss x ss samples per pixel, the first one at the pixel centre (depth and normal come from it)
    off = [((i + 0.5) / ss - 0.5) for i in range(ss)]
    off = [0.0] + [o for o in off if o != 0.0] if ss % 2 == 1 else off
    xs = torch.arange(W, device=dev).float() + 0.5
    ys = torch.arange(H, device=dev).float() + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    if ss > 1:
        oo = torch.tensor([(a, b) for b in off for a in off], device=dev)
        if ss % 2 == 0:   # put an exact centre sample first for depth / normals
            oo = torch.cat([torch.zeros(1, 2, device=dev), oo])
        xx = (xx[None] + oo[:, 0, None, None]).reshape(-1, W)
        yy = (yy[None] + oo[:, 1, None, None]).reshape(-1, W)
    nsamp = xx.numel() // (W * H)
    cam = torch.stack([(xx - Kt[0, 2]) / Kt[0, 0], (yy - Kt[1, 2]) / Kt[1, 1], torch.ones_like(xx)], -1).reshape(-1, 3)
    ln = cam.norm(dim=1, keepdim=True)
    d = (cam / ln) @ c2w_t[:3, :3].T
    o = c2w_t[:3, 3].expand_as(d).contiguous()
    cols, ts, ns = [], [], []
    for k in range(0, d.shape[0], chunk):
        c, t, _ = tracer.trace(o[k:k + chunk], d[k:k + chunk].contiguous(), shadows, haze=haze)
        cols.append(c); ts.append(t); ns.append(tracer.last_normal)
    c = torch.cat(cols).reshape(nsamp, H, W, 3)
    c = c[1:].mean(0) if (ss > 1 and ss % 2 == 0) else c.mean(0)
    t = torch.cat(ts)[: W * H]
    z = torch.where(torch.isinf(t), torch.zeros_like(t), t / ln.squeeze(1)[: W * H]).reshape(H, W)
    if normals:
        return c.pow(1 / 1.15).clamp(0, 1), z, torch.cat(ns)[: W * H].reshape(H, W, 3)
    return c.pow(1 / 1.15).clamp(0, 1), z


def add_haze(img, depth, alpha=None):
    """Distance haze applied after the fact (the reconstruction views are rendered without it)."""
    hz = (1 - torch.exp(-depth / HAZE_M))[..., None]
    h = HORIZON.to(img.device).pow(1 / 1.15)
    if alpha is not None:
        hz = hz * (alpha[..., None] > 0)
    return img * (1 - hz) + h * hz
