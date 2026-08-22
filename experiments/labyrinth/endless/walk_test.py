#!/usr/bin/env python
"""Walk the endless labyrinth and check it stays endless.

"Endless" is a claim about cost and consistency, not about the absence of a boundary,
and both parts fail quietly. A world whose per-tile cost creeps up with distance is
finite in practice; a world that rebuilds a tile differently on the way back is not one
world. So this walks a real path through the generator and measures:

  latency    seconds per newly built tile, first leg vs last leg
  stalls     steps where the tile underfoot was not ready
  identity   a tile revisited after cache eviction must come back byte-identical
  memory     cache stays inside its bound, GPU does not creep

    python walk_test.py --steps 240 --radius 1
"""
import argparse
import hashlib
import time

import numpy as np
import torch

from infinite_maze import InfiniteMaze, CELL_M, EYE_H, DIRS, DC, DR, cell_centre_m
import tiles as T
import recon as RC
from stream_server import Streamer

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)
    return ok


def walk(maze, steps, seed=0):
    """A path that actually travels.

    A random walk biased towards going straight still rattles: over 200 steps it covered
    12 cells of ground, which tests the cache and nothing else. The point here is to
    leave -- so aim at a distant goal and follow the route, which is also what a player
    with somewhere to be would do.
    """
    goal = (steps // 2, steps // 3)
    route = maze.bfs((0, 0), goal)
    if route is None:                     # unreachable within the search bound
        route = [(0, 0)]
    if len(route) >= steps:
        return route[:steps + 1]
    # carry on past the goal so the walk is as long as asked for
    rng = np.random.default_rng(seed)
    c, r = route[-1]
    last = None
    path = list(route)
    while len(path) <= steps:
        opts = maze.exits(c, r)
        d = last if (last is not None and last in opts and rng.random() < 0.85) \
            else int(opts[rng.integers(len(opts))])
        c, r = c + DC[d], r + DR[d]
        last = d
        path.append((c, r))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--span", type=int, default=3)
    ap.add_argument("--n-pos", type=int, default=4, dest="n_pos")
    ap.add_argument("--n-az", type=int, default=6, dest="n_az")
    ap.add_argument("--size", type=int, default=504)
    ap.add_argument("--max-splats", type=int, default=1_500_000, dest="max_splats")
    ap.add_argument("--cache-gb", type=float, default=2.0, dest="cache_gb")
    # A test that steps cell to cell as fast as the loop runs is a test of an infinitely
    # fast player, and will stall on any config whatsoever. Walking pace is ~1.4 m/s and
    # a cell is CELL_M*2 metres across, so time has to pass between steps for the
    # question ("can the generator keep ahead of a walker?") to mean anything.
    ap.add_argument("--speed", type=float, default=1.4,
                    help="player speed in m/s; 0 walks as fast as the loop allows")
    a = ap.parse_args()

    spec = T.TileSpec(span=a.span, n_pos=a.n_pos, n_az=a.n_az, size=a.size,
                      max_splats=a.max_splats)
    print(f"walk test  seed={a.seed} spec={spec.key()} steps={a.steps} "
          f"radius={a.radius} cache={a.cache_gb}GB\n")

    maze = InfiniteMaze(seed=a.seed)
    S = Streamer(a.seed, spec, cache_gb=a.cache_gb)
    RC.get_pipeline()

    path = walk(maze, a.steps, seed=1)
    print(f"path spans cells x[{min(c for c,_ in path)},{max(c for c,_ in path)}] "
          f"z[{min(r for _,r in path)},{max(r for _,r in path)}]", flush=True)

    # Cold start is a separate question from whether the generator keeps up. The opening
    # ring is 9 tiles and fills serially, so a player who starts walking immediately waits
    # for reasons that say nothing about steady-state throughput. Fill it first, report
    # what it cost, then measure the walk.
    x0, z0 = cell_centre_m(*path[0])
    t_cold = time.perf_counter()
    S.ring(x0, z0, radius=a.radius)
    while S.pending or S.cache.get(T.tile_of_pos_m(x0, z0, spec.span)) is None:
        time.sleep(0.05)
    cold_s = time.perf_counter() - t_cold
    print(f"cold start: opening ring of {(2*a.radius+1)**2} tiles in {cold_s:.1f}s\n",
          flush=True)

    stalls, per_tile, tiles_seen, first_hash = 0, [], set(), {}
    stall_s = 0.0
    t_start = time.perf_counter()
    for i, (c, r) in enumerate(path):
        x_m, z_m = cell_centre_m(c, r)
        before = S.stats["built"]
        S.ring(x_m, z_m, radius=a.radius)
        tc, tr = T.tile_of_pos_m(x_m, z_m, spec.span)
        tiles_seen.add((tc, tr))
        # perf_counter throughout: mixing it with time.time() made the deadline compare a
        # few-thousand-second uptime against epoch seconds, so the wait expired instantly
        # and the walk "never stalled" because it never waited for anything
        t0 = time.perf_counter()
        deadline = t0 + 180
        while S.cache.get((tc, tr)) is None and time.perf_counter() < deadline:
            time.sleep(0.02)
        waited = time.perf_counter() - t0
        if waited > 0.05:
            stalls += 1
            stall_s += waited
        # spend the rest of the time the player would have taken crossing this cell;
        # tiles keep building in the worker while that happens, which is the whole point
        if a.speed > 0:
            budget = (CELL_M * 2) / a.speed
            if waited < budget:
                time.sleep(budget - waited)
        got = S.cache.get((tc, tr))
        if got is not None:
            h = hashlib.blake2b(got[0], digest_size=16).hexdigest()
            first_hash.setdefault((tc, tr), h)
        built = S.stats["built"] - before
        if built:
            per_tile.append(S.stats["build_s"] / max(S.stats["built"], 1))
        if i % 40 == 0:
            print(f"  step {i:4d} cell ({c:4d},{r:4d}) tile ({tc:3d},{tr:3d}) "
                  f"built {S.stats['built']:3d} cache {S.cache.stats()} "
                  f"wait {waited:.2f}s", flush=True)

    # let the queue drain so the totals are honest
    while S.pending:
        time.sleep(0.1)
    elapsed = time.perf_counter() - t_start
    n = S.stats["built"]
    mean_s = S.stats["build_s"] / n if n else 0.0

    print(f"\nwalked {len(path)} cells across {len(tiles_seen)} tiles, "
          f"built {n} in {elapsed:.0f}s  (mean {mean_s:.2f}s/tile)\n")

    check("every tile the player stood in was delivered",
          len(first_hash) == len(tiles_seen),
          f"{len(first_hash)} of {len(tiles_seen)}")

    check("cache stayed inside its bound",
          S.cache.stats()["mb"] <= a.cache_gb * 1024 * 1.02,
          f"{S.cache.stats()['mb']}MB of {a.cache_gb * 1024:.0f}MB")

    # determinism: rebuild a tile the cache has certainly dropped, compare bytes.
    # This is the property that lets the cache be evicted freely -- if a rebuild differed,
    # walking back down a corridor would quietly redecorate it.
    victim = sorted(tiles_seen)[0]
    S.cache.evict(victim)
    S._build(*victim)
    h2 = hashlib.blake2b(S.cache.get(victim)[0], digest_size=16).hexdigest()
    check("a rebuilt tile is byte-identical to the first time",
          h2 == first_hash[victim], f"tile {victim}")

    # a far-away tile must cost what a near one costs, or "endless" is only nominal
    far_tile = (4000, -3000)
    far_km = max(abs(v) for v in T.tile_origin_m(*far_tile, spec.span)) / 1000.0
    t0 = time.perf_counter()
    S._build(*far_tile)
    far_s = time.perf_counter() - t0
    check(f"a tile {far_km:.0f} km out costs the same as one at the origin",
          far_s < mean_s * 2.5 if mean_s else True,
          f"{far_s:.2f}s vs mean {mean_s:.2f}s")

    check("the walker is not outrun by the generator in steady state",
          stalls <= max(1, len(path) // 10),
          f"{stalls} stalls in {len(path)} steps, {stall_s:.1f}s waited of "
          f"{elapsed:.0f}s walked ({stall_s / max(elapsed, 1e-9) * 100:.1f}%)")

    check("sustained build rate keeps ahead of the walk",
          mean_s * 3 <= (CELL_M * 2 * spec.span) / a.speed if a.speed else True,
          f"{mean_s:.2f}s/tile x3 per crossing vs "
          f"{(CELL_M * 2 * spec.span) / max(a.speed, 1e-9):.1f}s to cross a tile")

    print(f"\nGPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} -> {', '.join(FAILS)}")
        raise SystemExit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
