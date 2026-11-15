#!/usr/bin/env python
"""Pick the lighting by measurement.

The world came out unreadably dark, and the instinct is to raise gain in the viewer.
That would have been wrong: the reconstruction tracks the raycaster to 1.02x with 0.94
alpha coverage, so the viewer is faithfully reproducing a dark world. The darkness is in
the source, and anything the client does to it is stretching contrast that was never
captured.

Measured at the original settings, over a corridor view:

    median pixel 0.093, p90 0.105, max 0.212, nearest torch 8.04 m

Torches were 2.5 cells apart -- 10 m -- so with pure inverse-square the nearest one had
already fallen to 0.04 before shadowing took another 2.5x on top.

What "right" means here is a judgement, so it is stated as numbers rather than left to
taste: a torchlit vault should read as dim but legible, with real falloff between
torches rather than flat fill.

    median  0.25 - 0.45      the walls are readable
    p90     > 0.55           near a torch is genuinely bright
    p01     > 0.02           nothing is crushed to black
    max     > 0.85           the flames themselves still blow out

    python bench_light.py
"""
import itertools

import numpy as np
import torch

from infinite_maze import InfiniteMaze, EYE_H, cell_centre_grid, CELL_M
import render as R

TARGET = dict(median=(0.25, 0.45), p90=(0.55, 1.0), p01=(0.02, 1.0), mx=(0.85, 1.0))


def stats(spacing, cells=((1, 1), (4, 2), (7, 5), (2, 8)), size=(252, 189)):
    maze = InfiniteMaze(seed=1000, spacing_cells=spacing)
    vals, near = [], []
    for (c, r) in cells:
        win = R.build_window(maze, c - 8, r - 8, 17, 17)
        gx, gz = cell_centre_grid(c, r)
        if win.n_torch:
            cam = torch.tensor([gx - win.off_x, EYE_H, gz - win.off_z],
                               device=win.lights.device)
            near.append(float((win.lights - cam).norm(dim=1).min() * CELL_M))
        for yaw in (0.31, 1.9, 3.4, 5.0):
            img = R.render_view_tensor(win, (gx, EYE_H, gz), yaw, 0.0,
                                       width=size[0], height=size[1])
            vals.append(img.mean(0).flatten())
    v = torch.cat(vals)
    return {"median": float(v.median()), "p90": float(torch.quantile(v, 0.90)),
            "p01": float(torch.quantile(v, 0.01)), "mx": float(v.max()),
            "near_m": float(np.mean(near)) if near else float("nan")}


def score(s):
    """Distance outside the target band, summed. Zero means every criterion is met."""
    out = 0.0
    for k, (lo, hi) in TARGET.items():
        v = s[k]
        if v < lo:
            out += (lo - v) / max(lo, 1e-6)
        elif v > hi:
            out += (v - hi) / max(hi, 1e-6)
    return out


def main():
    print(f"target: {TARGET}\n")
    rows = []
    for spacing, k in itertools.product((2.5, 1.75, 1.25, 1.0), (0.62, 0.45, 0.34, 0.25)):
        R.FALLOFF_K = k
        s = stats(spacing)
        sc = score(s)
        rows.append((sc, spacing, k, s))
        print(f"  spacing {spacing:4.2f} cells ({spacing*2*CELL_M:5.1f} m)  falloff k={k:.2f}  "
              f"median {s['median']:.3f}  p90 {s['p90']:.3f}  p01 {s['p01']:.3f}  "
              f"max {s['mx']:.3f}  nearest {s['near_m']:5.2f}m   score {sc:.3f}",
              flush=True)
    rows.sort(key=lambda x: x[0])
    sc, spacing, k, s = rows[0]
    print(f"\n  best: spacing {spacing} cells, FALLOFF_K {k}  (score {sc:.3f})")
    print(f"        median {s['median']:.3f}  p90 {s['p90']:.3f}  "
          f"p01 {s['p01']:.3f}  max {s['mx']:.3f}")


if __name__ == "__main__":
    main()
