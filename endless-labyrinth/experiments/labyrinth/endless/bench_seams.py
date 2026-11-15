#!/usr/bin/env python
"""Is the join between two tiles worse than the middle of one?

bench_surface.py answers "how good is this tile", and it cannot answer this. Its truth
walks CELLS and samples each cell's own box, so the doorway strip between two cells is
never sampled -- and a tile seam falls exactly on such a strip. A context-camera layout
that left the z seams with no camera facing them scored 39.5% F there and looked normal.

So truth is rebuilt here from the occupancy grid itself: every free grid cell contributes
floor and ceiling, every free/solid boundary contributes a wall face. That covers
doorways, which is the whole point.

Two other things this gets right, both of which invert the answer if got wrong:

  the union, not the tile   A tile's crop stops 0.65 m short of its own boundary; the
                            neighbour covers the rest. Scoring one tile alone reports its
                            crop edge as a hole that a player never sees.
  inset from the box        Wall faces at exactly the tile-box edge belong to the
                            NEIGHBOUR's corridor. Scoring them against tiles that were
                            never meant to contain them reported "100% uncovered" for a
                            seam that is in fact solid wall.

    python bench_seams.py [--server http://127.0.0.1:8092] [--tiles 0,0 1,0 0,1]
"""
import argparse
import json
import sys
import urllib.request

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/zrx/HY-World-2.0")
sys.path.insert(0, "/home/zrx/HY-World-2.0/experiments/labyrinth/endless")

from infinite_maze import InfiniteMaze, CELL_M, WALL_H
from tiles import tile_origin_m

STEP = 0.06


def surface_points(maze, c0, r0, cols, rows):
    """Every surface the raycaster would draw over a cell box, doorways included."""
    occ = np.asarray(maze.occupancy(c0, r0, cols, rows))       # True = solid
    nz, nx = occ.shape
    m, hi = float(CELL_M), float(WALL_H * CELL_M)
    ax = lambda i: (2 * c0 + i) * m                            # noqa: E731
    az = lambda j: (2 * r0 + j) * m                            # noqa: E731
    out = []
    for j in range(1, nz - 1):
        for i in range(1, nx - 1):
            if occ[j, i]:
                continue
            x0, x1, z0, z1 = ax(i), ax(i + 1), az(j), az(j + 1)
            xs = np.arange(x0, x1, STEP)
            zs = np.arange(z0, z1, STEP)
            ys = np.arange(0.0, hi, STEP)
            gx, gz = np.meshgrid(xs, zs, indexing="ij")
            fx, fz = gx.ravel(), gz.ravel()
            for y in (0.0, hi):
                out.append(np.stack([fx, np.full_like(fx, y), fz], 1))
            for di, dj, px, pz in ((1, 0, x1, None), (-1, 0, x0, None),
                                   (0, 1, None, z1), (0, -1, None, z0)):
                if not occ[j + dj, i + di]:
                    continue                                   # open: no face here
                if px is not None:
                    gy, gzz = np.meshgrid(ys, zs, indexing="ij")
                    out.append(np.stack([np.full(gy.size, px), gy.ravel(), gzz.ravel()], 1))
                else:
                    gy, gxx = np.meshgrid(ys, xs, indexing="ij")
                    out.append(np.stack([gxx.ravel(), gy.ravel(), np.full(gy.size, pz)], 1))
    return np.concatenate(out)


def fetch(server, version, tc, tr):
    url = f"{server}/tile?tc={tc}&tr={tr}&wait=1&v={version}"
    r = urllib.request.urlopen(url, timeout=600)
    a = np.frombuffer(r.read(), dtype="<f4").reshape(-1, 14)
    ox, oz = json.loads(r.headers["X-Tile-Meta"])["origin_m"]
    p = a[:, :3].astype(np.float64).copy()
    p[:, 0] += ox
    p[:, 2] += oz
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8092")
    ap.add_argument("--tiles", nargs="*", default=["0,0", "1,0", "0,1", "1,1", "-1,-1"])
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args()

    spec = json.load(urllib.request.urlopen(a.server + "/spec"))
    span, version = spec["span"], spec["version"]
    maze = InfiniteMaze(seed=spec["seed"])
    pitch = 4.0 * span                       # metres between tile origins
    print(f"server {a.server}  world {version}  span {span}  tile pitch {pitch:.0f} m\n")

    agg = {}
    half = CELL_M * span                      # tile half-pitch: seams sit at ox +- half
    for t in a.tiles:
        ctc, ctr = (int(v) for v in t.split(","))
        P = np.concatenate([fetch(a.server, version, ctc + dc, ctr + dr)
                            for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
        tree = cKDTree(P)
        ox, oz = tile_origin_m(ctc, ctr, span)
        tp = surface_points(maze, ctc * span - 1, ctr * span - 1, span + 2, span + 2)
        eps = 0.01                            # drop the neighbour's own boundary faces
        tp = tp[(tp[:, 0] > ox - half - 1 + eps) & (tp[:, 0] < ox + half + 1 - eps)
                & (tp[:, 2] > oz - half - 1 + eps) & (tp[:, 2] < oz + half + 1 - eps)]
        d, _ = tree.query(tp, k=1, workers=-1)

        # A tile has FOUR seams, not two. This used to measure |x - (ox + half)| and
        # |z - (oz + half)| only, so the west and north joins were never labelled seam
        # at all -- they fell into the interior baseline, where a total hole along an
        # entire seam moved the printed number by 0.6 points against a 21-point
        # tile-to-tile spread. Deleting every splat within 0.35 m of the west seam of
        # tile (0,0) -- 61,905 of them -- moved the old x-seam figure by 0.0.
        dxw, dxe = tp[:, 0] - (ox - half), tp[:, 0] - (ox + half)
        dzn, dzs = tp[:, 2] - (oz - half), tp[:, 2] - (oz + half)
        sx = np.minimum(np.abs(dxw), np.abs(dxe))
        sz = np.minimum(np.abs(dzn), np.abs(dzs))
        print(f"tile ({ctc},{ctr})  truth {len(tp):,} pts  union {len(P):,} splats")

        # Per side as well as pooled: a defect that is one-sided -- and the rig has one,
        # the context cameras' pitch schedule aliases so the west camera never tilts
        # down and the east never tilts up -- cancels exactly when W and E are averaged.
        bands = [("x seam", sx < 0.35), ("z seam", sz < 0.35),
                 ("  seam W", np.abs(dxw) < 0.35), ("  seam E", np.abs(dxe) < 0.35),
                 ("  seam N", np.abs(dzn) < 0.35), ("  seam S", np.abs(dzs) < 0.35),
                 ("interior", (sx > 1.0) & (sz > 1.0)),
                 ("belt", ((sx >= 0.35) & (sx <= 1.0)) | ((sz >= 0.35) & (sz <= 1.0)))]
        for label, sel in bands:
            if sel.sum() < 30:
                print(f"  {label:9s} no surface in this band (a closed wall, not a hole)")
                continue
            dd = d[sel]
            near = 100 * (dd < a.tol).mean()
            far = 100 * (dd > 0.25).mean()
            print(f"  {label:9s} {sel.sum():7,} pts  median {np.median(dd) * 100:5.1f}cm  "
                  f"<{a.tol * 100:.0f}cm {near:5.1f}%  >25cm {far:5.1f}%")
            agg.setdefault(label, []).append((near, far))
        covered = int(((sx < 0.35) | (sz < 0.35) | ((sx > 1.0) & (sz > 1.0))
                       | ((sx >= 0.35) & (sx <= 1.0)) | ((sz >= 0.35) & (sz <= 1.0))).sum())
        assert covered == len(tp), f"bands do not partition the truth: {covered}/{len(tp)}"
        print()

    print("mean over tiles")
    for label, v in agg.items():
        v = np.array(v)
        print(f"  {label:9s} <{a.tol * 100:.0f}cm {v[:, 0].mean():5.1f}%   "
              f">25cm {v[:, 1].mean():5.1f}%   ({len(v)} tiles)")


if __name__ == "__main__":
    main()
