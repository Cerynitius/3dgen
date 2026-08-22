#!/usr/bin/env python
"""Adversarial geometry audit: compare splats against the surfaces we KNOW are there.

Every metric used so far can be gamed by removing geometry. depth_mae scores only
pixels where the reconstruction already rendered something, so discarding the splats it
was least sure about "improves" it -- confidence filtering at 25% reported 8.8 cm
against 11.2 cm while emptying 38% of the frame. edge energy went to 1.43x of ground
truth, which is impossible for real detail and is the speckle of holes.

The labyrinth's true surfaces are known analytically, so this scores against them and
reports BOTH directions of the Chamfer distance. They cannot be satisfied at once by
cheating:

  accuracy  splat -> nearest true surface. Deleting bad splats IMPROVES this.
  coverage  true surface -> nearest splat. Deleting anything MAKES IT WORSE.

A configuration is only better if it moves both.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/zrx/HY-World-2.0/experiments/labyrinth/endless")
from infinite_maze import CELL_M, WALL_H, DIRS, DC, DR
import tiles as T


def true_surface_points(maze, tc, tr, span, step=0.06, device="cuda", ring=0):
    """Sample the surfaces the raycaster would draw, in tile-local metres.

    `ring` extends the sampling by that many cells beyond the tile. It must be 1 for the
    ACCURACY direction and 0 for COVERAGE, and getting that wrong flatters or maligns a
    configuration for free. The crop keeps splats out to CELL_M*span+0.35 = 4.35 m while
    the tile's own cells end at 4.0 m, and a splat resting in a doorway sits on the
    NEIGHBOUR cell's floor. Sampling only the tile's own cells scores those splats against
    a surface that was never sampled, so they measure as far-from-truth when they are
    exactly right.

    KNOWN BLIND SPOT: this walks CELLS and samples each cell's own box, so the doorway
    strip BETWEEN two cells is never truth -- and a tile seam falls exactly on such a
    strip. Seam quality is therefore invisible here, which is how a context-camera
    placement that left the z seams with no camera facing them passed every audit in this
    file. Use bench_seams.py for that question: it builds truth from the occupancy grid
    (free cell -> floor/ceiling, free-solid boundary -> wall face), which covers doorways.
    The numbers here stay comparable across configs; they just do not answer it.
    """
    c0, r0 = T.tile_cells(tc, tr, span)
    ox, oz = T.tile_origin_m(tc, tr, span)
    hi = float(WALL_H * CELL_M)
    pts = []
    for r in range(r0 - ring, r0 + span + ring):
        for c in range(c0 - ring, c0 + span + ring):
            # open cell (c,r) spans [4c+2, 4c+4] x [4r+2, 4r+4] in world metres
            x0, x1 = 4 * c + 2.0, 4 * c + 4.0
            z0, z1 = 4 * r + 2.0, 4 * r + 4.0
            xs = np.arange(x0, x1, step)
            zs = np.arange(z0, z1, step)
            ys = np.arange(0.0, hi, step)
            gx, gz = np.meshgrid(xs, zs, indexing="ij")
            flat = np.stack([gx.ravel(), gz.ravel()], 1)
            for y in (0.0, hi):                      # floor and ceiling
                p = np.stack([flat[:, 0], np.full(len(flat), y), flat[:, 1]], 1)
                pts.append(p)
            for d in DIRS:                            # walls that are actually there
                if not maze.has_wall(c, r, d):
                    continue
                gy, gu = np.meshgrid(ys, xs if DC[d] == 0 else zs, indexing="ij")
                n = gy.size
                if DC[d] != 0:                        # wall on an x face
                    xf = x1 if DC[d] > 0 else x0
                    p = np.stack([np.full(n, xf), gy.ravel(), gu.ravel()], 1)
                else:                                 # wall on a z face
                    zf = z1 if DR[d] > 0 else z0
                    p = np.stack([gu.ravel(), gy.ravel(), np.full(n, zf)], 1)
                pts.append(p)
    P = np.concatenate(pts, 0)
    P[:, 0] -= ox
    P[:, 2] -= oz
    return torch.tensor(P, dtype=torch.float32, device=device)


def chamfer(a, b, chunk=4096):
    """min distance from every point of a to the set b, over the FULL set b.

    Subsampling b to a fixed COUNT silently makes this a measure of sampling density:
    widening the truth region from 2x2 cells to 4x4 while capping at 60k points quartered
    the density and moved measured accuracy from 33.3% to 27.7% with no change to the
    reconstruction at all. A KD-tree removes the need to subsample either side.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(b.detach().cpu().numpy())
    d, _ = tree.query(a.detach().cpu().numpy(), k=1, workers=-1)
    return torch.from_numpy(d).to(a.device).float()


def audit(g, maze, tc, tr, spec, tol=0.05, sample=60000):
    # accuracy scores splats against ALL nearby truth (ring=1); coverage asks only about
    # the surface this tile owns (ring=0)
    truth_acc = true_surface_points(maze, tc, tr, spec.span, device=g[0].device, ring=1)
    truth = true_surface_points(maze, tc, tr, spec.span, device=g[0].device, ring=0)
    # splats may be subsampled (they are interchangeable); the TRUTH sets never are,
    # because their density is the unit the distances are measured in
    m = g[0]
    if m.shape[0] > sample:
        m = m[torch.randperm(m.shape[0], device=m.device)[:sample]]
    d_sp = chamfer(m, truth_acc)  # splat -> surface : accuracy
    d_su = chamfer(truth, m)      # surface -> splat : coverage
    # precision / recall / F-score at tol, the Tanks-and-Temples formulation.
    # TransformerFusion (NeurIPS 2021) states the reason to combine them: "it is easy to
    # maximize either precision (by predicting only a few but accurate points) or recall
    # (by over-completing reconstructions with noisy surface), we found the most reliable
    # metric to be F-score". A 25%-confidence filter scoring 8.8 cm on 38% of the frame is
    # the precision half alone; F-score is what refuses it.
    prec = float((d_sp < tol).float().mean())
    rec = float((d_su < tol).float().mean())
    f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "acc_median_cm": float(d_sp.median()) * 100,
        "inlier_pct": prec * 100,
        "cov_median_cm": float(d_su.median()) * 100,
        "covered_pct": rec * 100,
        "fscore": f * 100,
        "splats": int(g[0].shape[0]),
        "truth_pts": int(truth.shape[0]),
    }
