#!/usr/bin/env python
"""Examine the two compositing knobs the web client was carrying unexamined.

`gain` and `dilate` live in the WebGL shader, which is why they were never swept: with
no browser there was nothing to score. But neither is graphics-specific -- both have
exact equivalents in the offline rasteriser, so they can be examined headlessly:

  dilate   splats.js adds u_dilate to cxx and cyy, the 2D covariance diagonal in px^2.
           That is precisely gsplat's eps2d, and gsplat's default rasterize_mode
           "classic" leaves opacity uncompensated for it exactly as the shader does.
  gain     covered = 1 - (1-alpha)^gain, a pure remap of per-splat opacity applied
           before any projection. Applying it to the opacity array is the same function.

What this does NOT prove is that the WebGL path agrees with gsplat pixel for pixel --
the two differ in quad bounds and cull thresholds. It settles the PARAMETERS, on the
same ground truth the rest of the pipeline is scored against.

Scored three ways, because PSNR alone chose blur once already:

  psnr    against the raycast, which is what the model was shown
  bright  rendered mean / ground-truth mean -- 1.0 means correctly exposed
  edge    rendered edge energy / ground truth's -- 1.0 means detail preserved, not
          smeared by dilation nor stippled by too little of it

    python bench_client.py
"""
import itertools

import numpy as np
import torch
from gsplat import rasterization

from infinite_maze import InfiniteMaze
import bench_config as B
import recon as RC
import tiles as T

CONV = RC.SplatConvention("wxyz", "sh", "raw", "none")


def render(g, cam, spec, gain, dilate):
    means, scales, quats, colors, opac = CONV.apply(g)
    if gain != 1.0:
        opac = (1.0 - (1.0 - opac).clamp(min=0) ** gain).clamp(0, 1)
    W, H = spec.wh()
    fx = (W / 2) / np.tan(spec.fov_x / 2)
    K = torch.tensor([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]],
                     dtype=torch.float32, device=means.device)
    img, _, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
        viewmats=CONV.viewmat(cam["c2w"], means.device).unsqueeze(0),
        Ks=K.unsqueeze(0), width=W, height=H, render_mode="RGB", eps2d=float(dilate))
    return img[0].clamp(0, 1)


def edge_energy(x):
    g = x.mean(-1)
    return float((g[:, 1:] - g[:, :-1]).abs().mean() + (g[1:] - g[:-1]).abs().mean())


def main():
    maze = InfiniteMaze(seed=1000)
    spec = T.TileSpec()
    print(f"config {spec.key()}\n")
    g, _, _ = B.build_block(maze, spec, n=2)
    cams, _ = B.block_cameras(maze, spec, n=2, per_kind=4)
    gt = [c["gt"].permute(1, 2, 0) for c in cams]
    gt_mean = float(np.mean([float(x.mean()) for x in gt]))
    gt_edge = float(np.mean([edge_energy(x) for x in gt]))
    print(f"ground truth: mean {gt_mean:.4f}, edge energy {gt_edge:.5f}\n")

    rows = []
    print("gain  dilate   psnr    bright   edge    score")
    for gain, dil in itertools.product((1.0, 1.2, 1.4, 1.8, 2.4, 3.2),
                                       (0.15, 0.30, 0.50, 0.80, 1.20)):
        ps, mn, eg = [], [], []
        for cam, ref in zip(cams, gt):
            im = render(g, cam, spec, gain, dil)
            ps.append(RC.psnr(im, ref))
            mn.append(float(im.mean()))
            eg.append(edge_energy(im))
        psnr = float(np.mean(ps))
        bright = float(np.mean(mn)) / gt_mean
        edge = float(np.mean(eg)) / gt_edge
        # exposure and detail are ratios that should both be 1; distance from 1 on each
        # is what a higher PSNR has to be worth. Weighted equally and in the same units.
        score = abs(np.log(max(bright, 1e-6))) + abs(np.log(max(edge, 1e-6)))
        rows.append((score, psnr, bright, edge, gain, dil))
        print(f"{gain:4.1f}  {dil:5.2f}   {psnr:5.2f}   {bright:5.2f}x  {edge:5.2f}x  {score:.3f}",
              flush=True)

    rows.sort(key=lambda r: r[0])
    print("\nclosest to correct exposure AND correct detail:")
    for sc, ps, br, ed, ga, di in rows[:5]:
        print(f"  gain {ga:.1f} dilate {di:.2f}   psnr {ps:5.2f}   "
              f"bright {br:.2f}x   edge {ed:.2f}x   score {sc:.3f}")
    best = rows[0]
    print(f"\n  -> gain={best[4]}, dilate={best[5]}")
    cur = [r for r in rows if r[4] == 1.4 and r[5] == 0.30]
    if cur:
        c = cur[0]
        print(f"  current client default gain=1.4 dilate=0.35: psnr {c[1]:.2f}, "
              f"bright {c[2]:.2f}x, edge {c[3]:.2f}x, score {c[0]:.3f}")


if __name__ == "__main__":
    main()
