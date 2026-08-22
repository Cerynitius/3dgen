#!/usr/bin/env python
"""Choose the streaming config by measurement.

Scoring a single tile is misleading twice over. A camera inside one tile sees past its
edge into territory the crop deliberately removed, so three quarters of the frame is
legitimately empty and the number says nothing. And a tile that reconstructs beautifully
on its own is worthless if it does not line up with the tile next to it -- which is the
one failure mode an endless world cannot tolerate, because the player crosses a seam
every few seconds.

So every configuration here is scored on an assembled BLOCK of tiles, placed by
arithmetic exactly as the client would place them, and photographed from cameras that
stand on the seams. The raycaster is the ground truth: it is what the views were drawn
from, so any disagreement is the reconstruction's.

    python bench_config.py --stage all
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from infinite_maze import InfiniteMaze, CELL_M, EYE_H, WALL_H, cell_centre_grid
import render as R
import recon as RC
import tiles as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench")


# ---------------------------------------------------------------- block assembly
def build_block(maze, spec, n=2, origin=(0, 0), conv=None, verbose=False):
    """Reconstruct an n x n block of tiles and place them in one world frame."""
    parts, metas, t_build = [], [], 0.0
    for j in range(n):
        for i in range(n):
            tc, tr = origin[0] + i, origin[1] + j
            t0 = time.perf_counter()
            g, meta, _ = RC.build_tile(maze, tc, tr, spec)
            t_build += time.perf_counter() - t0
            ox, oz = T.tile_origin_m(tc, tr, spec.span)
            m = g[0].clone()
            m[:, 0] += ox
            m[:, 2] += oz
            parts.append([m] + list(g[1:]))
            metas.append(meta)
            if verbose:
                print(f"    tile ({tc},{tr}) {meta['splats']:,} splats "
                      f"{meta['timing']['total']:.2f}s", flush=True)
    merged = [torch.cat([p[k] for p in parts], 0) for k in range(5)]
    return merged, metas, t_build


def block_cameras(maze, spec, n=2, origin=(0, 0), device=R.DEV, per_kind=4):
    """Ground-truth cameras inside the block: some in the middle of a tile, some
    standing exactly on a seam between tiles."""
    span = spec.span
    c0 = origin[0] * span
    r0 = origin[1] * span
    cells = n * span
    win = R.build_window(maze, c0 - spec.view_margin, r0 - spec.view_margin,
                         cells + 2 * spec.view_margin, cells + 2 * spec.view_margin,
                         device=device)
    W, H = spec.wh()

    interior = [(c0 + span // 2, r0 + span // 2),
                (c0 + span + span // 2, r0 + span // 2),
                (c0 + span // 2, r0 + span + span // 2),
                (c0 + span + span // 2, r0 + span + span // 2)]
    # A seam camera stands in the last cell of one tile facing the next -- but its yaw
    # used to come from the loop index (17.8 deg + i*90) with no reference to which
    # boundary it was standing on, so not one of the four ever photographed the join it
    # was placed to watch, and psnr_seam scored whatever happened to be in front of it.
    # Aim each along its own outward normal. yaw is atan2(dx, dz), as in render_rig.
    seam = [((c0 + span - 1, r0 + span // 2), np.pi / 2),      # last cell, looking +x
            ((c0 + span, r0 + span // 2), -np.pi / 2),         # first of the next, -x
            ((c0 + span // 2, r0 + span - 1), 0.0),            # looking +z
            ((c0 + span // 2, r0 + span), np.pi)]              # looking -z

    out = []
    for kind, cellset in (("interior", interior), ("seam", seam)):
        for i in range(per_kind):
            if kind == "seam":
                (cc, rr), aim = cellset[i % len(cellset)]
                # spread extra cameras about the normal instead of round the circle
                yaw = aim + 0.35 * (i // len(cellset) - 0.5)
            else:
                cc, rr = cellset[i % len(cellset)]
                yaw = 2 * np.pi * (i / per_kind) + 0.31
            gx, gz = cell_centre_grid(cc, rr)
            pitch = 0.16 * np.sin(3.0 * yaw)
            gt = R.render_view_tensor(win, (gx, EYE_H, gz), float(yaw), float(pitch),
                                      width=W, height=H, fov_x=spec.fov_x)
            depth = R.render_depth_tensor(win, (gx, EYE_H, gz), float(yaw), float(pitch),
                                          width=W, height=H, fov_x=spec.fov_x)
            # world frame: floor at y=0, so the eye sits one eye-height above it
            m4 = R.c2w(float(yaw), float(pitch),
                       (gx * CELL_M, EYE_H * CELL_M, gz * CELL_M))
            out.append({"gt": gt, "depth": depth, "c2w": m4, "kind": kind,
                        "cell": (cc, rr), "yaw": float(yaw), "pitch": float(pitch)})
    return out, win


def score_block(g, cams, spec, conv, dump=None):
    W, H = spec.wh()
    fx = (W / 2) / np.tan(spec.fov_x / 2)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=np.float32)
    by_kind, cov_all, rows = {}, [], []
    for i, cam in enumerate(cams):
        img = RC.rasterize(g, cam["c2w"], K, W, H, conv)
        gt = cam["gt"].permute(1, 2, 0)
        p = RC.psnr(img, gt)
        cov = float((img.sum(-1) > 0.01).float().mean())
        by_kind.setdefault(cam["kind"], []).append(p)
        cov_all.append(cov)
        rows.append((cam["kind"], p, cov))
        if dump:
            os.makedirs(dump, exist_ok=True)
            side = torch.cat([gt, img], 1).cpu().numpy()
            Image.fromarray((side * 255).astype(np.uint8)).save(
                os.path.join(dump, f"{cam['kind']}_{i:02d}.png"))
    return {"psnr": float(np.mean([p for _, p, _ in rows])),
            "psnr_interior": float(np.mean(by_kind.get("interior", [0]))),
            "psnr_seam": float(np.mean(by_kind.get("seam", [0]))),
            "coverage": float(np.mean(cov_all))}


# ---------------------------------------------------------------- stages
def stage_calibrate(maze, args):
    """Verify the splat convention against a camera the model was actually given.

    Novel views are the wrong place to settle a convention: at a held-out pose every
    candidate renders something wrong, and on a scene this dark PSNR is then decided by
    mean brightness rather than by alignment -- which is how a camera pointing backwards
    came out ahead twice. Scored instead against a rig view, where the correct
    convention has to reproduce the exact image the model was fed, the answer is
    unambiguous and can then be trusted at held-out poses.
    """
    print("\n=== stage 1: splat convention ===", flush=True)
    spec = T.TileSpec(span=3, n_pos=4, n_az=6, size=504)
    batch, extr, intr, _, _, _ = T.render_rig(maze, 0, 0, spec)
    pred, _ = RC.infer(batch, extr, intr)
    g, meta = T.postprocess(pred, spec, extr)
    W, H = spec.wh()
    K = intr[0, 0].cpu().numpy()
    gt0 = batch[0, 0].permute(1, 2, 0).to(R.DEV)
    # opacity is pinned to raw: act_gs.reg_dense_opacities is already a sigmoid, so the
    # values are probabilities and re-applying it would be a second activation. Its
    # softening effect is available honestly as TileSpec.opacity_gain, and swept below.
    best, top = None, -1e9
    for cam in ("none", "flipY", "flipZ", "flipYZ"):
        for quat in ("wxyz", "xyzw"):
            conv = RC.SplatConvention(quat, "sh", "raw", cam)
            img = RC.rasterize(g, extr[0, 0].cpu().numpy(), K, W, H, conv)
            p = RC.psnr(img, gt0)
            cov = float((img.sum(-1) > 0.01).float().mean())
            if p > top:
                best, top = conv, p
            print(f"    {p:6.2f} dB  coverage {cov * 100:5.1f}%   {conv}", flush=True)
    print(f"\n  chosen: {best}   ({top:.2f} dB reproducing a rig view)")
    print(f"  alignment residual {meta['align_rmse_m']} m; floor lands at "
          f"{meta['floor_err']} m and the ceiling within {meta['ceil_err']} m of the "
          f"known {WALL_H * CELL_M:.2f} m storey -- neither was fitted")
    return best


def stage_sweep(maze, args, conv, grid, label, base=None):
    print(f"\n=== {label} ===", flush=True)
    rows = []
    for cfg in grid:
        spec = T.TileSpec(**{**(base or {}), **cfg})
        torch.cuda.reset_peak_memory_stats()
        try:
            g, metas, t_build = build_block(maze, spec, n=args.block)
            cams, _ = block_cameras(maze, spec, n=args.block, per_kind=args.cams)
            s = score_block(g, cams, spec, conv)
        except torch.cuda.OutOfMemoryError:
            print(f"  {spec.key():28s}  OOM")
            torch.cuda.empty_cache()
            continue
        n_tiles = args.block ** 2
        # The number that decides whether a config can be streamed at all. Walking in a
        # straight line with a ring of radius R, crossing one tile means building the
        # 2R+1 tiles of the new column. If that takes longer than the crossing does, the
        # player outruns the generator and the world ends in front of them -- so this is
        # the walking speed the config can actually sustain, in m/s.
        tile_w = spec.span * 2 * CELL_M
        walk_mps = tile_w / ((2 * args.radius + 1) * (t_build / n_tiles))
        row = {**cfg, "key": spec.key(), "max_walk_mps": round(walk_mps, 2),
               "psnr": round(s["psnr"], 2),
               "psnr_interior": round(s["psnr_interior"], 2),
               "psnr_seam": round(s["psnr_seam"], 2),
               "seam_drop": round(s["psnr_interior"] - s["psnr_seam"], 2),
               "coverage": round(s["coverage"], 3),
               "splats_per_tile": int(np.mean([m["splats"] for m in metas])),
               "s_per_tile": round(t_build / n_tiles, 3),
               "views_s": round(np.mean([m["timing"]["views"] for m in metas]), 3),
               "fwd_s": round(np.mean([m["timing"]["forward"] for m in metas]), 3),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        rows.append(row)
        print(f"  {spec.key():28s} PSNR {row['psnr']:5.2f} "
              f"(int {row['psnr_interior']:5.2f} / seam {row['psnr_seam']:5.2f}, "
              f"drop {row['seam_drop']:+.2f})  cov {row['coverage']*100:4.1f}%  "
              f"{row['splats_per_tile']:>9,} splats  {row['s_per_tile']:5.2f}s/tile  "
              f"walk {row['max_walk_mps']:5.2f}m/s  {row['peak_gb']:4.1f}GB", flush=True)
        del g
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--block", type=int, default=2, help="tiles per side in the block")
    ap.add_argument("--cams", type=int, default=4, help="cameras per kind")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--radius", type=int, default=1,
                    help="ring radius the walk-speed estimate assumes")
    ap.add_argument("--fix-pos", type=int, default=0, dest="fix_pos")
    ap.add_argument("--fix-az", type=int, default=0, dest="fix_az")
    ap.add_argument("--fix-size", type=int, default=0, dest="fix_size")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    maze = InfiniteMaze(seed=a.seed)
    allrows = {}

    conv = stage_calibrate(maze, a)
    allrows["convention"] = repr(conv)
    rig = {"n_pos": a.fix_pos or 4, "n_az": a.fix_az or 6}
    size = a.fix_size or 504
    span = 3

    if a.stage in ("all", "rig"):
        grid = [{"n_pos": p, "n_az": z} for p, z in
                ((1, 8), (2, 8), (4, 4), (4, 6), (4, 8), (4, 12), (9, 4), (9, 6))]
        allrows["rig"] = stage_sweep(maze, a, conv, grid,
                                     "stage 2: rig geometry (positions x azimuths)",
                                     base={"span": span, "size": size})
        b = max(allrows["rig"], key=lambda r: r["psnr"])
        rig = {"n_pos": b["n_pos"], "n_az": b["n_az"]}
        print(f"\n  best rig: {b['key']}  {b['psnr']:.2f} dB")

    if a.stage in ("all", "size"):
        grid = [{"size": p} for p in (364, 504, 644, 952)]
        allrows["size"] = stage_sweep(maze, a, conv, grid, "stage 3: view resolution",
                                      base={"span": span, **rig})
        size = max(allrows["size"], key=lambda r: r["psnr"])["size"]

    if a.stage in ("all", "span"):
        grid = [{"span": s} for s in (2, 3, 4)]
        allrows["span"] = stage_sweep(maze, a, conv, grid, "stage 4: tile span",
                                      base={"size": size, **rig})
        span = max(allrows["span"], key=lambda r: r["psnr"])["span"]

    if a.stage in ("all", "opacity"):
        grid = [{"opacity_gain": o} for o in (1.0, 0.8, 0.6, 0.45, 0.3)]
        allrows["opacity"] = stage_sweep(maze, a, conv, grid,
                                         "stage 5: opacity gain (redundancy softening)",
                                         base={"size": size, "span": span, **rig})
        gain = max(allrows["opacity"], key=lambda r: r["psnr"])["opacity_gain"]
    else:
        gain = 1.0

    if a.stage in ("all", "budget"):
        grid = [{"max_splats": m} for m in (0, 1_500_000, 800_000, 400_000, 200_000)]
        allrows["budget"] = stage_sweep(maze, a, conv, grid,
                                        "stage 6: splat budget per tile",
                                        base={"size": size, "span": span,
                                              "opacity_gain": gain, **rig})

    with open(os.path.join(OUT, f"results_{a.stage}.json"), "w") as f:
        json.dump(allrows, f, indent=1)
    print(f"\nwrote {os.path.join(OUT, f'results_{a.stage}.json')}")


if __name__ == "__main__":
    main()
