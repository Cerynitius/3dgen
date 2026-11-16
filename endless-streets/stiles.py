"""Street tiles: the unit the town is reconstructed and streamed in.

A tile is a TILE x TILE metre square centred on (TILE*a, TILE*b). The street network is irregular, so the
lattice does not follow it: a tile is built when a street corridor crosses it, its cameras stand on the crossings
and street segments that reach into it, and the squares inside the blocks (never walked) are skipped, their area
going to the nearest built tile.

A tile is built like a walk through a filmed street: a rig of eye-level cameras on the
walkable area is rendered by the ray tracer (these are the tile's "photos", kept for the
fixer as reference views), the feed-forward model turns them into Gaussians, and the
Gaussians are put into metres with the rig's exact poses and one scale per view fitted on
the true depth, cropped to the tile and stripped of splats that lie off every real surface.

Postprocessing reuses the measured choices of the endless labyrinth (see its CONFIG.md).
"""
import math
import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

import raytrace as RT
import world as Wd

LAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "endless-labyrinth", "experiments", "labyrinth",
                   "endless")    # sibling checkout (3dgen repo layout, or ~/endless-streets next to ~/endless-labyrinth)
sys.path.insert(0, LAB)
import tiles as LT  # noqa: E402  (labyrinth helpers: scale per view, unprojection, pruning)

TILE = 20.0


@dataclass
class Spec:
    size: int = 476            # square views, 14-px patches
    fov_x: float = math.radians(100)
    n_az: int = 6
    pitch: float = 0.10
    up_pitch: float = 0.62
    max_splats: int = 600_000
    surface_reject_cm: float = 10.0
    # ... or this fraction of the camera distance, whichever is larger: the model's depth error grows with
    # distance, and a fixed 10 cm deleted 19-26% of the points, mostly real upper floors seen from the street
    # (holes 16% -> 1.6% of a tile's pixels at 3%, measured 11-15)
    surface_reject_rel: float = 0.03
    reject_rel_from: float = 0.0   # metres: the relative tolerance counts camera distance beyond this only
    # per voxel keep points from views at most this x farther than the nearest (0 = off). 50 cm / 2x measured
    # 11-15 on two tiles: holes 2.23% -> 1.83% of surface pixels and no far-view speckle on near facades
    near_wins: float = 2.0
    near_wins_voxel: float = 0.5
    scale_quantile: float = 0.98
    margin: float = 0.0
    ss: int = 2                  # supersampling of the reconstruction views
    down_pitch: float = -0.55   # one view per position pitched at the ground (0 = none)
    positions: str = "truth"     # "truth": unproject the ray tracer's depth; "model": gs_depth x per-view scale
    sky_mask: bool = True        # drop splats of pixels that see sky
    appearance: str = "pixel"    # "pixel": splat colour = the view's own pixel; "model": the model's colour
    bake: bool = False           # True: no model at all -- one surface-aligned disc per pixel
    bake_sigma: float = 0.45     # disc std-dev in pixel footprints
    bake_voxel: float = 0.02     # metres; one splat per voxel, from the view that sees it largest
    # holes: after the tile is built it is rendered from walker-like probe views (eye height, every probe_step m on
    # its streets, 8 headings, level and tilted up) at low resolution; pixels it leaves uncovered whose ray hits a
    # surface this tile owns get surface-aligned discs from the ray tracer, so holes cost nothing while walking
    patch_holes: bool = True
    probe_step: float = 3.0
    probe_w: int = 256
    probe_h: int = 144
    probe_alpha: float = 0.5


@lru_cache(maxsize=65536)
def built(a, b, seed=0):
    """A tile is built when a street corridor (carriageway or sidewalk) crosses its square."""
    o, h = origin(a, b), TILE / 2
    x0, x1, z0, z1 = o[0] - h, o[0] + h, o[2] - h, o[2] + h
    for s in Wd.segments_in(x0, x1, z0, z1, seed):
        if s["fam"] == 0:
            hit = s["a0"] < x1 and s["a1"] > x0 and s["c"] - s["hw"] < z1 and s["c"] + s["hw"] > z0
        else:
            hit = s["a0"] < z1 and s["a1"] > z0 and s["c"] - s["hw"] < x1 and s["c"] + s["hw"] > x0
        if hit:
            return True
    return False


def kind(a, b, seed=0):
    return "street" if built(a, b, seed) else "interior"


def owned(pw, a, b, seed=0):
    """Which points (world xz) tile (a, b) keeps: those in its square, and those in squares that are not built
    (block interiors) whose nearest built square (distance from the point, within two squares) is this one --
    otherwise the backs of the buildings and the side walls above lower neighbours are never reconstructed."""
    A = torch.round(pw[:, 0] / TILE).long()
    B = torch.round(pw[:, 2] / TILE).long()
    keep = torch.zeros(pw.shape[0], dtype=torch.bool, device=pw.device)
    near = ((A - a).abs() <= 2) & ((B - b).abs() <= 2)
    if not near.any():
        return keep
    R = 4                                           # table of built squares around (a, b)
    tab = torch.tensor([[built(a + i, b + j, seed) for j in range(-R, R + 1)] for i in range(-R, R + 1)],
                       device=pw.device)
    idx = torch.nonzero(near).squeeze(1)
    Ai, Bi = A[idx] - a + R, B[idx] - b + R
    own = tab[Ai, Bi]
    keep[idx[own & (Ai == R) & (Bi == R)]] = True
    q = torch.nonzero(~own).squeeze(1)
    if q.numel():
        px, pz = pw[idx[q], 0], pw[idx[q], 2]
        best = torch.full((q.numel(),), float("inf"), device=pw.device)
        arg = torch.full((q.numel(), 2), -99, dtype=torch.long, device=pw.device)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ci, cj = Ai[q] + di, Bi[q] + dj
                ok = (ci >= 0) & (ci <= 2 * R) & (cj >= 0) & (cj <= 2 * R)
                b_ok = ok & tab[ci.clamp(0, 2 * R), cj.clamp(0, 2 * R)]
                cx, cz = (ci - R + a).float() * TILE, (cj - R + b).float() * TILE
                dx = ((px - cx).abs() - TILE / 2).clamp(min=0)
                dz = ((pz - cz).abs() - TILE / 2).clamp(min=0)
                dd = torch.where(b_ok, dx * dx + dz * dz, torch.full_like(dx, float("inf")))
                better = dd < best
                best = torch.where(better, dd, best)
                arg[better] = torch.stack([ci[better], cj[better]], -1)
        keep[idx[q[(arg[:, 0] == R) & (arg[:, 1] == R)]]] = True
    return keep


def origin(a, b):
    return np.array([a * TILE, 0.0, b * TILE], np.float32)


def rig(a, b, spec=Spec(), seed=0):
    """Camera poses (c2w, tile frame) of tile (a, b): level rings on the walkable area of every crossing and street
    segment that reaches into the tile, views tilted up at the facades, and context cameras further along the
    streets looking in. Cameras that would stand off the streets (a T-junction's missing arm) are dropped."""
    E = Wd.EYE
    o = origin(a, b)
    h = TILE / 2
    x0, x1, z0, z1 = o[0] - h, o[0] + h, o[2] - h, o[2] + h
    azs = [2 * math.pi * i / spec.n_az for i in range(spec.n_az)]
    cams = []

    def add(x, z, yaw, pitch):                     # world position -> tile frame
        if Wd.on_street(x, z, seed):
            cams.append(RT.c2w(yaw, pitch, (x - o[0], E, z - o[2])))
    near_x = []
    for i, j, X, Z in Wd.crossings_in(x0 - 11, x1 + 11, z0 - 11, z1 + 11, seed):
        hx, hz = Wd.lhw(1, i, seed), Wd.lhw(0, j, seed)
        if min(X + hx, x1) - max(X - hx, x0) < 3.0 or min(Z + hz, z1) - max(Z - hz, z0) < 3.0:
            continue                                # its junction box barely reaches into the tile
        near_x.append((i, j))
        sx, sz = Wd.lroad(1, i, seed) + 1.3, Wd.lroad(0, j, seed) + 1.3
        pos = [(0.0, 0.0)] + [(qx * sx, qz * sz) for qx in (-1, 1) for qz in (-1, 1)]
        for px, pz in pos:
            for y in azs:
                add(X + px, Z + pz, y + (0.5 if (px, pz) != (0.0, 0.0) else 0) * (2 * math.pi / spec.n_az), spec.pitch)
        if spec.down_pitch:
            for n, (px, pz) in enumerate(pos):
                for q in range(2):
                    add(X + px, Z + pz, n * 1.1 + q * math.pi, spec.down_pitch)
        for qx in (-1, 1):
            for qz in (-1, 1):      # up at the corner facades across the junction
                add(X + qx * sx, Z + qz * sz, math.atan2(-qx, -qz), spec.up_pitch)
                add(X - qx * 1.0, Z - qz * 1.0, math.atan2(qx, qz), spec.up_pitch)
        for (px, pz), y in (((TILE * 0.8, 0), -math.pi / 2), ((-TILE * 0.8, 0), math.pi / 2),
                            ((0, TILE * 0.8), math.pi), ((0, -TILE * 0.8), 0.0)):
            add(X + px, Z + pz, y, spec.pitch)
    for sg in Wd.segments_in(x0, x1, z0, z1, seed):
        fam, c, road, hw = sg["fam"], sg["c"], sg["road"], sg["hw"]
        if not (c - hw < (z1 if fam == 0 else x1) and c + hw > (z0 if fam == 0 else x0)):
            continue
        # along-range of this segment inside the tile, outside the junction boxes
        _, _, _, b0, b1 = Wd.seg_geom(fam, sg["k"], sg["m"], seed)
        lo, hi = (x0, x1) if fam == 0 else (z0, z1)
        lo, hi = max(lo, b0), min(hi, b1)
        if hi - lo < 1.0:
            continue
        ends = [(sg["m"], sg["k"]), (sg["m"] + 1, sg["k"])] if fam == 0 else [(sg["k"], sg["m"]), (sg["k"], sg["m"] + 1)]
        if hi - lo < 8.0 and any(e in near_x for e in ends):
            continue                                # a short arm next to a junction this tile already films
        alongx = fam == 0
        s = road + (1.3 if road > 3 else 0.65)          # sidewalk line

        def at(u, w):
            return (u, c + w) if alongx else (c + w, u)
        n = 1 if hi - lo < 6 else (2 if hi - lo < 12 else 3)
        us = [(lo + hi) / 2] if n == 1 else [lo + 1.5 + (hi - lo - 3) * q / (n - 1) for q in range(n)]
        ws = [lo + (hi - lo) * 0.25, lo + (hi - lo) * 0.75] if hi - lo >= 6 else [(lo + hi) / 2]
        pos = [(u, 0.0) for u in us] + [(u, sd * s) for u in ws for sd in (-1, 1)]
        for u, w in pos:
            for q in range(4):
                y = q * math.pi / 2 + (math.pi / 4 if w != 0 else 0)
                add(*at(u, w), y, spec.pitch)
        if spec.down_pitch:
            for q, (u, w) in enumerate(pos):
                add(*at(u, w), q * 1.3 + (0 if alongx else math.pi / 2), spec.down_pitch)
        for u in us:                    # up at both facades from the far kerb
            for sd in (-1, 1):
                yaw = (0.0 if sd > 0 else math.pi) if alongx else (math.pi / 2 if sd > 0 else -math.pi / 2)
                add(*at(u, -sd * s), yaw, spec.up_pitch)
        # context: further along the street, looking back into the tile
        # (forward is (sin yaw, cos yaw): +x is yaw pi/2, +z is yaw 0)
        for u, sgn, dw, tilt in ((hi + 0.35 * TILE, -1, 0.0, 0.0), (lo - 0.35 * TILE, 1, 0.0, 0.0),
                                 (hi + 0.2 * TILE, -1, 2.0, -0.25), (lo - 0.2 * TILE, 1, -2.0, 0.25)):
            base = (math.pi / 2 if sgn > 0 else -math.pi / 2) if alongx else (0.0 if sgn > 0 else math.pi)
            add(*at(u, dw), base + tilt, spec.pitch)
    if not cams:                                    # a tile only grazed by a corridor: a ring at its street point
        for y in azs:
            cams.append(RT.c2w(y, spec.pitch, (0.0, E, 0.0)))
    return np.stack(cams)


def tile_scene(a, b, reach=90.0, seed=0, device="cuda"):
    o = origin(a, b)
    return RT.Scene(o[0] - reach, o[0] + reach, o[2] - reach, o[2] + reach, seed=seed, device=device)


@torch.no_grad()
def render_rig(a, b, spec=Spec(), seed=0, device="cuda", scene=None):
    """-> batch [1,N,3,S,S], c2w_tile [N,4,4], K [3,3], depths [N,S,S] (metres, 0 = sky), scene."""
    # only geometry near the tile matters: pixels whose surface lies outside the footprint are cropped, and the
    # tallest building (~30 m) casts a shadow ~24 m long at this sun elevation
    sc = scene or tile_scene(a, b, reach=35.0, seed=seed, device=device)
    tr = RT.Raytracer(sc)
    o = origin(a, b)
    c2ws = rig(a, b, spec, seed)
    K = RT.K_of(spec.size, spec.size, spec.fov_x)
    imgs, deps, nors = [], [], []
    for m in c2ws:
        w = m.copy()
        w[:3, 3] += o
        img, z, nn = RT.render(tr, w, K, spec.size, spec.size, normals=True, haze=False, ss=spec.ss)
        imgs.append(img.permute(2, 0, 1))
        deps.append(z)
        nors.append(nn)
    render_rig.normals = torch.stack(nors)
    return torch.stack(imgs)[None], c2ws, K, torch.stack(deps), sc


def build(a, b, pipe_infer, spec=Spec(), seed=0, device="cuda"):
    """One tile end to end -> splats (tile frame, metres) as a dict of tensors, and a report."""
    t = {}
    t0 = time.perf_counter()
    batch, c2ws, K, depths, sc = render_rig(a, b, spec, seed, device)
    t["views"] = time.perf_counter() - t0
    extr = torch.from_numpy(c2ws)[None]
    intr = torch.from_numpy(np.stack([K] * len(c2ws)))[None]
    t1 = time.perf_counter()
    S_, HH, WW = depths.shape
    pix = batch[0].permute(0, 2, 3, 1).reshape(-1, 3).to(device)
    if spec.bake:
        fake = {"gs_depth": depths[None, ..., None].to(device)}
        means = LT.means_from_true_cameras(fake, extr, intr, 1.0)
        nrm = render_rig.normals.reshape(-1, 3).to(device)
        fx = float(K[0, 0])
        # disc radius = the pixel's footprint on the surface: z / fx / |cos(view, normal)|, capped
        cdir = means - torch.from_numpy(c2ws[:, :3, 3]).to(device).repeat_interleave(HH * WW, 0)
        cdir = torch.nn.functional.normalize(cdir, dim=-1)
        cosv = (cdir * nrm).sum(-1).abs().clamp(min=0.25)
        r = depths.reshape(-1).to(device) / fx
        s1 = (r * spec.bake_sigma)[:, None]
        scales = torch.cat([s1 / cosv[:, None], s1, (r * 0.08)[:, None]], 1)
        # rotation whose z axis is the normal; x axis along the view direction projected on the surface
        zax = nrm
        xax = torch.nn.functional.normalize(cdir - (cdir * zax).sum(-1, keepdim=True) * zax + 1e-6, dim=-1)
        yax = torch.cross(zax, xax, dim=-1)
        Rm = torch.stack([xax, yax, zax], -1)
        quats = LT.mat_to_quat_batch(Rm) if hasattr(LT, "mat_to_quat_batch") else _quat_from_mats(Rm)
        g = [means, scales, quats, ((pix - 0.5) / 0.28209479177387814), torch.full((means.shape[0], 1), 0.95, device=device)]
        n_raw = means.shape[0]
        sv = torch.ones(S_)
        g.append(r / cosv)   # true footprint on the surface (grazing views stretch it), picks the sharpest splat per voxel
        t["forward"] = 0.0
    else:
        pred, fwd = pipe_infer(batch, extr, intr)
        t["forward"] = fwd
        g = list(LT.gaussians_from_predictions(pred))
        n_raw = g[0].shape[0]
        dl = [(i, depths[i]) for i in range(depths.shape[0])]
        sv = LT.scale_per_view(pred, dl, len(c2ws))
        per = sv.to(g[1].device).repeat_interleave(HH * WW)[:, None]
        if spec.positions == "truth":
            fake = {"gs_depth": depths[None, ..., None].to(g[0].device)}
            g[0] = LT.means_from_true_cameras(fake, extr, intr, 1.0)
        else:
            g[0] = LT.means_from_true_cameras(pred, extr, intr, sv)
        g[1] = g[1] * per
        if spec.appearance == "pixel":
            g[3] = (pix.to(g[3].device) - 0.5) / 0.28209479177387814
    t2 = time.perf_counter()
    g = [x.to(device) for x in g]
    camd = depths.reshape(-1).to(device)
    if spec.sky_mask:
        m = camd > 0
        g = [x[m] for x in g]
        camd = camd[m]
    o = torch.from_numpy(origin(a, b)).to(device)
    m = owned(g[0] + o, a, b, seed) & (g[0][:, 1] > -0.3) & (g[0][:, 1] < 60)
    g = [x[m] for x in g]
    camd = camd[m]
    n_crop = int(m.sum())
    # off-surface rejection against the analytic town geometry, primitives near the owned area only (baked splats
    # sit on the ray tracer's own depth, so they are on a surface by construction: nothing to reject)
    n_float = 0
    if not spec.bake:
        h = 2.5 * TILE + 1
        near = RT.Scene(float(o[0]) - h, float(o[0]) + h, float(o[2]) - h, float(o[2]) + h, seed=seed, device=device)
        dist = torch.cat([near.surface_distance(g[0][k:k + 200_000] + o) for k in range(0, g[0].shape[0], 200_000)])
        keep = dist <= torch.clamp(spec.surface_reject_rel * (camd - spec.reject_rel_from), min=spec.surface_reject_cm / 100)
        n_float = int((~keep).sum())
        g = [x[keep] for x in g]
        camd = camd[keep]
    if spec.near_wins and not spec.bake:
        # where a close view saw a surface, drop what far views put there: the looser far-view tolerance above
        # is for surfaces only far views reach (upper floors), not for speckling near facades
        q = torch.floor(g[0] / spec.near_wins_voxel).long()
        q -= q.min(0).values
        mm = q.max(0).values + 1
        _, inv = torch.unique((q[:, 0] * mm[1] + q[:, 1]) * mm[2] + q[:, 2], return_inverse=True)
        cmin = torch.full((int(inv.max()) + 1,), float("inf"), device=camd.device).scatter_reduce(0, inv, camd, "amin")
        nw = camd <= spec.near_wins * cmin[inv]
        g = [x[nw] for x in g]
    if spec.bake:
        g, n_big = _sharpest_per_voxel(g, spec.bake_voxel)
        vox = spec.bake_voxel
    else:
        g, n_big = LT.scale_quantile_filter(g, spec.scale_quantile)
        g, vox = LT.voxel_prune(g, spec.max_splats, merge="global", grow=True, pick="opacity")
    t["post"] = time.perf_counter() - t2
    n_patch = 0
    if spec.patch_holes:
        t3 = time.perf_counter()
        g, n_patch = patch_holes(g, a, b, spec, seed, device, sc)
        t["patch"] = time.perf_counter() - t3
    t["total"] = time.perf_counter() - t0
    means, scales, quats, colors, opac = g
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    out = {"means": means.float(), "scales": scales.float(), "quats": quats.float(),
           "rgb": (0.28209479177387814 * colors + 0.5).clamp(0, 1).float(), "opacity": opac.reshape(-1).clamp(0, 1).float(),
           "rig_c2w": torch.from_numpy(c2ws), "rig_K": torch.from_numpy(K)}
    rep = {"tile": [a, b], "kind": kind(a, b, seed), "views": len(c2ws), "raw": n_raw, "cropped": n_crop,
           "floaters": n_float, "oversized": n_big, "patched": n_patch, "splats": int(means.shape[0]),
           "scale_med": round(float(sv.median()), 4), "timing": {k: round(v, 3) for k, v in t.items()}}
    return out, rep


def probes(a, b, spec=Spec(), seed=0):
    """Walker-like camera poses (c2w, tile frame) on the tile's streets: eye height, a grid of probe_step m,
    8 headings, level and tilted up."""
    o, h = origin(a, b), TILE / 2
    st = spec.probe_step
    cams = []
    for x in np.arange(o[0] - h + st / 2, o[0] + h, st):
        for z in np.arange(o[2] - h + st / 2, o[2] + h, st):
            if not Wd.on_street(float(x), float(z), seed):
                continue
            for q in range(8):
                for pitch in (0.05, 0.4):
                    cams.append(RT.c2w(q * math.pi / 4 + 0.2 * (pitch > 0.1), pitch, (x - o[0], Wd.EYE, z - o[2])))
    return np.stack(cams) if cams else np.zeros((0, 4, 4), np.float32)


@torch.no_grad()
def patch_holes(g, a, b, spec, seed, device, sc):
    """Fill the holes walker-like views see in this tile. g = [means, scales, quats, sh colours, opacities] (tile
    frame) -> (g with the patch splats appended, number of patch splats)."""
    from gsplat import rasterization
    cams = probes(a, b, spec, seed)
    if not len(cams) or not g[0].shape[0]:
        return g, 0
    W, H = spec.probe_w, spec.probe_h
    K = torch.tensor(RT.K_of(W, H, math.radians(90)), device=device)
    fx = float(K[0, 0])
    means, scales, quats, cols, opac = g
    sc_r = scales.clone()
    if spec.bake:              # as drawn while walking (server.py): a disc spans at least its voxel
        sc_r[:, :2] = sc_r[:, :2].clamp(min=0.45 * spec.bake_voxel)
    rgb = (0.28209479177387814 * cols + 0.5).clamp(0, 1)
    q = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ys, xs = torch.meshgrid(torch.arange(H, device=device) + 0.5, torch.arange(W, device=device) + 0.5, indexing="ij")
    cam = torch.nn.functional.normalize(torch.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1],
                                                     torch.ones_like(xs)], -1).reshape(-1, 3), dim=-1)
    c2w = torch.from_numpy(cams).to(device)
    vms = torch.linalg.inv(c2w)
    o = torch.from_numpy(origin(a, b)).to(device)
    tr = RT.Raytracer(sc)
    P_, N_, C_, R_, D_ = [], [], [], [], []
    for k in range(0, len(cams), 32):
        _, al, _ = rasterization(means, q, sc_r, opac.reshape(-1), rgb, vms[k:k + 32],
                                 K[None].expand(min(32, len(cams) - k), 3, 3), W, H, packed=False)
        hole = al[..., 0].reshape(al.shape[0], -1) < spec.probe_alpha
        for v in range(al.shape[0]):
            m = hole[v]
            if not m.any():
                continue
            d = (cam[m] @ c2w[k + v, :3, :3].T).contiguous()
            ro = (c2w[k + v, :3, 3] + o).expand_as(d).contiguous()
            # intersection is cheap, shading is not: find the uncovered rays that hit a surface this tile owns
            # (not the sky, not other tiles' surfaces) and shade only those
            t, _ = sc.intersect(ro, d)
            hit = torch.isfinite(t)
            if not hit.any():
                continue
            hi = torch.nonzero(hit).squeeze(1)
            p = ro[hi] + d[hi] * t[hi][:, None]
            own = owned(p, a, b, seed)
            if not own.any():
                continue
            hi = hi[own]
            c, t2, _ = tr.trace(ro[hi].contiguous(), d[hi].contiguous(), shadows=True, haze=False)
            ok = torch.isfinite(t2)
            hi, c, t2 = hi[ok], c[ok], t2[ok]
            P_.append(ro[hi] + d[hi] * t2[:, None] - o)
            N_.append(tr.last_normal[ok])
            C_.append(c.pow(1 / 1.15))
            R_.append(t2 * cam[m][hi, 2] / fx)               # pixel footprint at the probe resolution
            D_.append(d[hi])
    if not P_:
        return g, 0
    p, nrm, c, r, d = (torch.cat(x) for x in (P_, N_, C_, R_, D_))
    cosv = (d * nrm).sum(-1).abs().clamp(min=0.25)
    # probe pixels are coarser than rig pixels: discs wide enough to close up at the probe's pixel spacing
    s1 = torch.maximum(r * 0.6, torch.full_like(r, 0.5 * spec.bake_voxel))[:, None]
    sc_new = torch.cat([s1 / cosv[:, None], s1, (r * 0.08)[:, None]], 1)
    xax = torch.nn.functional.normalize(d - (d * nrm).sum(-1, keepdim=True) * nrm + 1e-6, dim=-1)
    Rm = torch.stack([xax, torch.cross(nrm, xax, dim=-1), nrm], -1)
    q_new = LT.mat_to_quat_batch(Rm) if hasattr(LT, "mat_to_quat_batch") else _quat_from_mats(Rm)
    new = [p, sc_new, q_new, (c - 0.5) / 0.28209479177387814, torch.full((p.shape[0], 1), 0.95, device=device),
           r / cosv]
    new, _ = _sharpest_per_voxel(new, spec.bake_voxel)          # many probes see the same hole
    return [torch.cat([x, y.to(x.dtype).reshape(-1, *x.shape[1:])]) for x, y in zip(g, new)], int(new[0].shape[0])


def _sharpest_per_voxel(g, vs):
    """Keep, in every voxel of size vs, the splat with the smallest pixel footprint (the view closest to it)."""
    means, r = g[0], g[-1]
    q = torch.floor(means / vs).long()
    q -= q.min(0).values
    m = q.max(0).values + 1
    key = (q[:, 0] * m[1] + q[:, 1]) * m[2] + q[:, 2]
    order = torch.argsort(r)
    key_o = key[order]
    so = torch.argsort(key_o, stable=True)
    ks = key_o[so]
    first = torch.ones_like(ks, dtype=torch.bool)
    first[1:] = ks[1:] != ks[:-1]
    keep = order[so[first]]
    return [x[keep] for x in g[:-1]], int(means.shape[0] - keep.numel())


def _quat_from_mats(R):
    """[N,3,3] rotation matrices -> [N,4] wxyz quaternions (Shepperd, batched)."""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = torch.stack([
        torch.stack([1 + tr, R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1),
        torch.stack([R[:, 2, 1] - R[:, 1, 2], 1 + m00 - m11 - m22, R[:, 0, 1] + R[:, 1, 0], R[:, 0, 2] + R[:, 2, 0]], -1),
        torch.stack([R[:, 0, 2] - R[:, 2, 0], R[:, 0, 1] + R[:, 1, 0], 1 - m00 + m11 - m22, R[:, 1, 2] + R[:, 2, 1]], -1),
        torch.stack([R[:, 1, 0] - R[:, 0, 1], R[:, 0, 2] + R[:, 2, 0], R[:, 1, 2] + R[:, 2, 1], 1 - m00 - m11 + m22], -1)], 1)
    k = torch.stack([tr, m00, m11, m22], -1).argmax(-1)
    q = q[torch.arange(R.shape[0], device=R.device), k]
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def load_pipeline():
    import recon as LR
    LR.WEIGHTS = os.path.expanduser("~/models/HY-World-2.0")
    LR.get_pipeline()
    return LR.infer
