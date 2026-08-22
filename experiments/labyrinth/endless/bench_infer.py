#!/usr/bin/env python
"""Time and A/B the WorldMirror forward across the fastinfer layers.

Run once per configuration (the layers are env flags read at import), saving the
packed tile bytes and timings; run again with --compare to diff configurations on
identical inputs. Tiles are chosen to have different view counts, because the view
count varies with the maze and any optimisation has to survive that.

  LAB_SKIP_HEADS=0 LAB_ROPE_CACHE=0 LAB_SYNC_FREE=0 python bench_infer.py --save base.pkl
  python bench_infer.py --compare base.pkl                      # default fast layers
  LAB_COMPILE=1 python bench_infer.py --compare base.pkl
  LAB_COMPILE=1 LAB_CUDAGRAPH=1 python bench_infer.py --compare base.pkl
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch

import recon as RC
import tiles as T
from infinite_maze import InfiniteMaze


def rig_views(maze, tc, tr, spec):
    span = spec.span
    c0, r0 = tc * span, tr * span
    n = spec.n_pos * spec.n_az + spec.n_pos + spec.n_context * spec.n_az_context
    for cc, rr in [(c0 + i, r0 + j) for j in range(span) for i in range(span)]:
        for dN, dE in ((0, 1), (2, 1), (2, 3), (0, 3)):
            if maze.has_wall(cc, rr, dN) and maze.has_wall(cc, rr, dE):
                n += 1
    return n


def find_tiles_by_views(maze, spec, want):
    seen = {}
    for r in range(-6, 6):
        for c in range(-6, 6):
            occ = maze.occupancy(c * spec.span - 1, r * spec.span - 1,
                                 spec.span + 2, spec.span + 2)
            if float(occ.mean()) > 0.92:
                continue
            n = rig_views(maze, c, r, spec)
            if n not in seen:
                seen[n] = (c, r)
            if len(seen) >= want:
                return sorted(seen.items())
    return sorted(seen.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--tiles", type=int, default=3)
    ap.add_argument("--save", default=None)
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()

    import fastinfer as FI
    label = " ".join(k for k, v in FI.FLAGS.items() if v) or "baseline"
    maze = InfiniteMaze(seed=a.seed)
    spec = T.TileSpec()
    picks = find_tiles_by_views(maze, spec, a.tiles)
    print(f"[bench] config: {label}")
    print(f"[bench] view counts: {[n for n, _ in picks]}")

    RC.get_pipeline()
    results = {}
    for n_views, (tc, tr) in picks:
        batch, extr, intr, win, _, depths = T.render_rig(maze, tc, tr, spec)
        times = []
        for it in range(a.repeats + 1):
            pred, dt = RC.infer(batch, extr, intr)
            if it:                                    # skip warmup/capture pass
                times.append(dt)
            if it != a.repeats:
                del pred
        fwd = float(np.median(times))
        t1 = time.perf_counter()
        g, meta = T.postprocess(pred, spec, extr, intr, depths, maze, tc, tr)
        torch.cuda.synchronize()
        post = time.perf_counter() - t1
        payload, n = T.pack_splats(g)
        results[(tc, tr)] = {"views": int(batch.shape[1]), "forward_s": fwd,
                             "post_s": post, "bytes": payload, "meta": meta}
        print(f"  tile ({tc:3d},{tr:3d})  {batch.shape[1]:2d} views  "
              f"forward {fwd:6.3f}s  post {post:5.3f}s  splats {n:,}  "
              f"floor {meta['floor_err']:+.3f}")
        del pred, g
        torch.cuda.empty_cache()
    print(f"[bench] peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GB")

    if a.save:
        with open(a.save, "wb") as f:
            pickle.dump({"label": label, "results": results}, f)
        print(f"[bench] saved -> {a.save}")

    if a.compare:
        with open(a.compare, "rb") as f:
            ref = pickle.load(f)
        print(f"[bench] against '{ref['label']}':")
        for key, r in results.items():
            b = ref["results"].get(key)
            if b is None:
                continue
            same = r["bytes"] == b["bytes"]
            if same:
                verdict = "bytes IDENTICAL"
            else:
                x = np.frombuffer(r["bytes"], "<f4").reshape(-1, 14)
                y = np.frombuffer(b["bytes"], "<f4").reshape(-1, 14)
                if x.shape == y.shape:
                    d = np.abs(x - y)
                    verdict = (f"splats equal-count; max|d| pos "
                               f"{d[:, :3].max():.2e} scale {d[:, 3:6].max():.2e} "
                               f"col {d[:, 10:13].max():.2e}")
                else:
                    verdict = f"splat count {x.shape[0]:,} vs {y.shape[0]:,}"
            print(f"  tile {key}: {b['forward_s']:.3f}s -> {r['forward_s']:.3f}s "
                  f"({b['forward_s']/r['forward_s']:.2f}x)  {verdict}")


if __name__ == "__main__":
    main()
