"""Score a tile against the ray tracer on held-out walker views, on the pixels whose true surface lies
inside the tile's footprint (so missing neighbours do not count against it)."""
import math

import numpy as np
import torch

import raytrace as RT
import splatrender as SR
import stiles as ST


def held_out(a, b, n=12, seed=1):
    rng = np.random.default_rng(seed * 1000 + a * 37 + b)
    k = ST.kind(a, b)
    o = ST.origin(a, b)
    out = []
    while len(out) < n:
        x, z = rng.uniform(-ST.TILE / 2, ST.TILE / 2, 2)
        lx, lz = x + o[0], z + o[2]
        mx = abs((lx + 20) % 40 - 20); mz = abs((lz + 20) % 40 - 20)
        if not (mx < 5.5 or mz < 5.5):
            continue
        out.append(RT.c2w(rng.uniform(-math.pi, math.pi), rng.uniform(-0.25, 0.3), (lx, rng.uniform(1.3, 1.9), lz)))
    return out


@torch.no_grad()
def score(g, a, b, tracer, W=640, H=360, fov=math.radians(90)):
    o = ST.origin(a, b)
    G = SR.cat_tiles([(g, o)])
    K = RT.K_of(W, H, fov)
    Kt = torch.tensor(K).cuda()
    ps, de, cov = [], [], []
    for c in held_out(a, b):
        gt, z = RT.render(tracer, c, K, W, H, haze=False, ss=2)
        ct = torch.tensor(c).cuda()
        xs = torch.arange(W, device="cuda") + 0.5
        ys = torch.arange(H, device="cuda") + 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        cam = torch.stack([(xx - K[0, 2]) / K[0, 0] * z, (yy - K[1, 2]) / K[1, 1] * z, z], -1)
        pw = cam.reshape(-1, 3) @ ct[:3, :3].T + ct[:3, 3]
        m = ((pw[:, 0] - o[0]).abs() <= ST.TILE / 2) & ((pw[:, 2] - o[2]).abs() <= ST.TILE / 2) & (z.reshape(-1) > 0)
        m = m.reshape(H, W)
        if m.sum() < 2000:
            continue
        img, dep, al = SR.render(G, torch.linalg.inv(ct), Kt, W, H)
        mse = ((img - gt) ** 2).mean(-1)[m].mean()
        ps.append(-10 * math.log10(float(mse)))
        ok = m & (dep > 0)
        de.append(float(((dep - z).abs() / z.clamp(min=0.1))[ok].median()))
        cov.append(float((al[m] > 0.5).float().mean()))
    return {"psnr": round(float(np.mean(ps)), 2), "depth_rel": round(float(np.mean(de)), 4), "cover": round(float(np.mean(cov)), 3), "n": len(ps)}
