"""Street tiles: the unit the town is reconstructed and streamed in.

A tile is a TILE x TILE metre square centred on (TILE*a, TILE*b). With TILE = P/2 the lattice puts a
crossing at every (even, even) tile, a street segment at every (odd, even) / (even, odd) tile, and
block interiors (never walked, never seen from the street) at (odd, odd), which are skipped.

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

import numpy as np
import torch

import raytrace as RT
import world as Wd

LAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "endless-labyrinth", "experiments", "labyrinth",
                   "endless")    # sibling checkout (3dgen repo layout, or ~/endless-streets next to ~/endless-labyrinth)
sys.path.insert(0, LAB)
import tiles as LT  # noqa: E402  (labyrinth helpers: scale per view, unprojection, pruning)

TILE = Wd.P / 2


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


def kind(a, b):
    ea, eb = a % 2 == 0, b % 2 == 0
    if ea and eb:
        return "crossing"
    if ea or eb:
        return "street_z" if ea else "street_x"   # street_x: the road runs along x through the tile
    return "interior"


def owned(pw, a, b):
    """Which points (world xz) tile (a, b) keeps. Interior tiles (block centres) are never built, so their area
    goes to the nearest street tile -- otherwise the back half of every building and the side walls above lower
    neighbours are never reconstructed."""
    A = torch.round(pw[:, 0] / TILE).long()
    B = torch.round(pw[:, 2] / TILE).long()
    inter = (A % 2 != 0) & (B % 2 != 0)
    dx, dz = pw[:, 0] - A * TILE, pw[:, 2] - B * TILE
    zs = dz.abs() > dx.abs()
    B = torch.where(inter & zs, B + torch.sign(dz).long(), B)
    A = torch.where(inter & ~zs, A + torch.sign(dx).long(), A)
    return (A == a) & (B == b)


def origin(a, b):
    return np.array([a * TILE, 0.0, b * TILE], np.float32)


def rig(a, b, spec=Spec()):
    """Camera poses (c2w, tile frame) of tile (a, b): level rings on the walkable area, views tilted up at
    the facades, and context cameras in the neighbouring tiles looking in."""
    k = kind(a, b)
    E = Wd.EYE
    cams = []
    azs = [2 * math.pi * i / spec.n_az for i in range(spec.n_az)]
    s = Wd.ROAD + 1.3        # sidewalk line
    if k == "crossing":
        pos = [(0, 0)] + [(sx * s, sz * s) for sx in (-1, 1) for sz in (-1, 1)]
        for x, z in pos:
            for i, y in enumerate(azs):
                cams.append(RT.c2w(y + (0.5 if (x, z) != (0, 0) else 0) * (2 * math.pi / spec.n_az), spec.pitch, (x, E, z)))
        if spec.down_pitch:
            for j, (x, z) in enumerate(pos):
                for q in range(2):
                    cams.append(RT.c2w(j * 1.1 + q * math.pi, spec.down_pitch, (x, E, z)))
        for sx in (-1, 1):
            for sz in (-1, 1):      # up at the corner facades across the junction
                cams.append(RT.c2w(math.atan2(-sx, -sz), spec.up_pitch, (sx * s, E, sz * s)))
                cams.append(RT.c2w(math.atan2(sx, sz) , spec.up_pitch, (-sx * 1.0, E, -sz * 1.0)))
        ctx = [((TILE * 0.8, 0), -math.pi / 2), ((-TILE * 0.8, 0), math.pi / 2), ((0, TILE * 0.8), math.pi), ((0, -TILE * 0.8), 0.0)]
    else:
        along_x = k == "street_x"
        line = [-6.0, 0.0, 6.0]
        pos = [(u, 0.0) for u in line] + [(u, sd * s) for u in (-5.0, 5.0) for sd in (-1, 1)]
        for u, w in pos:
            x, z = (u, w) if along_x else (w, u)
            for i in range(4):
                y = i * math.pi / 2 + (math.pi / 4 if w != 0 else 0)
                cams.append(RT.c2w(y, spec.pitch, (x, E, z)))
        if spec.down_pitch:
            for j, (u, w) in enumerate(pos):
                x, z = (u, w) if along_x else (w, u)
                cams.append(RT.c2w(j * 1.3 + (0 if along_x else math.pi / 2), spec.down_pitch, (x, E, z)))
        for u in (-6.0, 0.0, 6.0):      # up at both facades from the far kerb
            for sd in (-1, 1):
                x, z = (u, -sd * s) if along_x else (-sd * s, u)
                yaw = (0.0 if sd > 0 else math.pi) if along_x else (math.pi / 2 if sd > 0 else -math.pi / 2)
                cams.append(RT.c2w(yaw, spec.up_pitch, (x, E, z)))
        if along_x:
            ctx = [((TILE * 0.85, 0), -math.pi / 2), ((-TILE * 0.85, 0), math.pi / 2),
                   ((TILE * 0.7, 2.0), -math.pi / 2 - 0.25), ((-TILE * 0.7, -2.0), math.pi / 2 - 0.25)]
        else:
            ctx = [((0, TILE * 0.85), math.pi), ((0, -TILE * 0.85), 0.0),
                   ((2.0, TILE * 0.7), math.pi + 0.25), ((-2.0, -TILE * 0.7), 0.25)]
    for (x, z), y in ctx:
        cams.append(RT.c2w(y, spec.pitch, (x, E, z)))
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
    c2ws = rig(a, b, spec)
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
    m = owned(g[0] + o, a, b) & (g[0][:, 1] > -0.3) & (g[0][:, 1] < 60)
    g = [x[m] for x in g]
    camd = camd[m]
    n_crop = int(m.sum())
    # off-surface rejection against the analytic town geometry, primitives near the owned area only
    h = TILE + 1
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
    t["total"] = time.perf_counter() - t0
    means, scales, quats, colors, opac = g
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    out = {"means": means.float(), "scales": scales.float(), "quats": quats.float(),
           "rgb": (0.28209479177387814 * colors + 0.5).clamp(0, 1).float(), "opacity": opac.reshape(-1).clamp(0, 1).float(),
           "rig_c2w": torch.from_numpy(c2ws), "rig_K": torch.from_numpy(K)}
    rep = {"tile": [a, b], "kind": kind(a, b), "views": len(c2ws), "raw": n_raw, "cropped": n_crop,
           "floaters": n_float, "oversized": n_big, "splats": int(means.shape[0]),
           "scale_med": round(float(sv.median()), 4), "timing": {k: round(v, 3) for k, v in t.items()}}
    return out, rep


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
