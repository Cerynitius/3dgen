#!/usr/bin/env python
"""Endless 3DGS labyrinth, streamed by where the player is standing.

The world is not generated; it is *resolved*. Every tile is a pure function of
(seed, tile index, spec), so a tile the player walks back to is the tile they left --
byte for byte, with no state kept to make that true. That is what lets the cache be
thrown away under pressure and the world still be the same world.

    python stream_server.py [--port 8092] [--seed 1000]

    GET /health                          model residency, queue depth, timings
    GET /ring?x=&z=&r=                   tiles around a position: ready / queued
    GET /tile?tc=&tr=                    binary splats (header X-Tile-Meta: json)
    GET /maze?c0=&r0=&cols=&rows=        wall bits, so the client can collide locally
    GET /spec                            the metric constants the client places by

A tile is generated once and then served from cache. Requests are served in order of
distance from the player, with the tile they are standing in always first, so the
frontier fills outward from where it is needed rather than in index order.
"""
import argparse
import heapq
import json
import os
import sys
import threading
import time
import hashlib
import math
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from infinite_maze import InfiniteMaze, CELL_M, WALL_H, EYE_H, DIRS
import tiles as T
import recon as RC
import render as R2


class TileCache:
    """Bounded LRU of packed tiles. Bounded by bytes, not by count, because a tile's
    size varies with how much geometry its cells actually contain."""

    def __init__(self, max_bytes=6 << 30):
        self.max_bytes = max_bytes
        self.bytes = 0
        self._d = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
            return v

    def put(self, key, payload, meta):
        with self._lock:
            if key in self._d:
                return
            self._d[key] = (payload, meta)
            self.bytes += len(payload)
            while self.bytes > self.max_bytes and len(self._d) > 1:
                _, (p, _m) = self._d.popitem(last=False)
                self.bytes -= len(p)

    def evict(self, key):
        """Drop one tile. Used by the walk test to prove a rebuilt tile comes back
        identical, which is only meaningful if the accounting stays straight."""
        with self._lock:
            got = self._d.pop(key, None)
            if got is not None:
                self.bytes -= len(got[0])
            return got is not None

    def stats(self):
        with self._lock:
            return {"tiles": len(self._d), "mb": round(self.bytes / 2**20, 1)}


class Streamer:
    def __init__(self, seed, spec, cache_gb=6, workers=2):
        self.maze = InfiniteMaze(seed=seed)
        # everything that changes what a tile looks like, hashed into one short token
        # Config alone is not enough. The opacity-compensation fix in voxel_prune changed
        # every tile without touching a single parameter, so the version stayed put and a
        # browser would have gone on serving the translucent ones from its year-long
        # cache. Hash the source of the modules that decide tile content as well, and any
        # code change busts it automatically.
        src = ""
        import fastinfer as FI
        for mod in (T, R2, __import__("infinite_maze"), RC, FI):
            try:
                with open(mod.__file__, "rb") as fh:
                    src += hashlib.blake2b(fh.read(), digest_size=8).hexdigest()
            except OSError:
                pass
        stamp = json.dumps({"seed": seed, "spec": T.asdict(spec),
                            "spacing": self.maze.spacing, "src": src,
                            "fast": FI.stamp(),
                            "light": [R2.REACH_M, R2.INTENSITY, R2.FALLOFF_K,
                                      R2.AMBIENT, R2.FILL]}, sort_keys=True)
        self.version = hashlib.blake2b(stamp.encode(), digest_size=6).hexdigest()
        self.seed = seed
        self.spec = spec
        self.cache = TileCache(int(cache_gb * (1 << 30)))
        self.pending = {}                 # key -> priority
        self.building = set()             # keys a worker currently holds
        self.heap = []
        self.cv = threading.Condition()
        self.stats = {"built": 0, "served": 0, "build_s": 0.0, "last": None}
        self.last_centre = None            # last /ring player position, metres
        self._bake_thread = threading.Thread(target=self._bake_loop, daemon=True)
        self._bake_thread.start()
        # One worker, measured. The reasoning for two was that a tile is not one
        # indivisible cost -- rendering its 24 views takes 0.73s and holds no model lock,
        # the forward pass takes 1.02s and holds it -- so pipelining should have given
        # max(0.73, 1.02) instead of 1.76s. It does not. Measured on a straight walk,
        # two workers need 7.23s per 8 m crossing against one worker's 5.49s: 24% SLOWER.
        # The model lock was never the constraint; the GPU is already saturated by a
        # single build, so a second one only contends for it.
        self.workers = [threading.Thread(target=self._loop, daemon=True)
                        for _ in range(max(1, workers))]
        for w in self.workers:
            w.start()

    # -- queue
    # Four times the largest legal ring (r=4 -> 81 tiles). Past this the queue holds
    # work for tiles no player is near, and since nothing is ever cancelled it would
    # stay there: accepting it is worse than refusing it, because the client's own
    # backoff can respond to a refusal and cannot respond to a silent 83-hour backlog.
    MAX_PENDING = 4 * 81

    def request(self, tc, tr, priority):
        key = (tc, tr)
        if self.cache.get(key) is not None:
            return "ready"
        with self.cv:
            if key not in self.pending and len(self.pending) >= self.MAX_PENDING:
                return "deferred"
            if key in self.pending:
                if priority < self.pending[key]:
                    self.pending[key] = priority
                    heapq.heappush(self.heap, (priority, key))
                return "queued"
            self.pending[key] = priority
            heapq.heappush(self.heap, (priority, key))
            self.cv.notify()
        return "queued"

    def _loop(self):
        while True:
            with self.cv:
                while not self.heap:
                    self.cv.wait()
                _, key = heapq.heappop(self.heap)
                if key not in self.pending:
                    continue
                self.pending.pop(key, None)
            if self.cache.get(key) is not None:
                continue
            with self.cv:
                if key in self.building:      # another worker already has it
                    continue
                self.building.add(key)
            try:
                self._build(*key)
            except Exception:
                traceback.print_exc()
            finally:
                with self.cv:
                    self.building.discard(key)

    def _build(self, tc, tr):
        t0 = time.perf_counter()
        # A tile that is almost all solid rock has too little surface for the
        # alignment SVD to fit -- it dies with "SVD did not converge", and the
        # queue then retries it forever. Answer definitively instead: an empty
        # tile is the truth for solid ground, cached so nobody asks twice.
        span = self.spec.span
        c0, r0 = tc * span, tr * span
        occ = self.maze.occupancy(c0 - 1, r0 - 1, span + 2, span + 2)
        if float(occ.mean()) > 0.92:
            self.cache.put((tc, tr), b"", {"tile": [tc, tr], "bytes": 0, "empty": True,
                                           "origin_m": list(T.tile_origin_m(tc, tr, span))})
            with self.cv:
                self.stats["built"] += 1
            print(f"[tile] ({tc},{tr}) solid -- empty tile, cached")
            return
        g, meta, _ = RC.build_tile(self.maze, tc, tr, self.spec)
        ox, oz = T.tile_origin_m(tc, tr, self.spec.span)
        meta["origin_m"] = [ox, oz]
        payload, n = T.pack_splats(g)
        meta["bytes"] = len(payload)
        del g
        self.cache.put((tc, tr), payload, meta)
        dt = time.perf_counter() - t0
        with self.cv:                       # two workers share these counters
            self.stats["built"] += 1
            self.stats["build_s"] += dt
            self.stats["last"] = meta["timing"]
        print(f"[tile] ({tc},{tr}) {n:,} splats  {dt:.2f}s  "
              f"align_rmse {meta['align_rmse_m']}m  floor {meta['floor_err']}m  "
              f"cache {self.cache.stats()}", flush=True)

    # -- the ring
    def ring(self, x_m, z_m, radius=1):
        """Tiles within `radius` of the player, nearest first.

        The tile underfoot is priority zero: a player who turns a corner into an
        ungenerated tile should wait on that tile and nothing else."""
        tc0, tr0 = T.tile_of_pos_m(x_m, z_m, self.spec.span)
        self.last_centre = (x_m, z_m)
        out = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                tc, tr = tc0 + dc, tr0 + dr
                d = (dc * dc + dr * dr) ** 0.5
                state = "ready" if self.cache.get((tc, tr)) is not None else None
                if state is None:
                    state = self.request(tc, tr, d)
                ox, oz = T.tile_origin_m(tc, tr, self.spec.span)
                out.append({"tile": [tc, tr], "state": state,
                            "origin_m": [ox, oz], "d": round(d, 3)})
        out.sort(key=lambda r: r["d"])
        return {"centre": [tc0, tr0], "span": self.spec.span, "tiles": out}


    def _bake_loop(self):
        """Idle pre-bake: when the queue is empty, generate the halo two rings beyond
        the player's last position, outer ring first so the nearer ring lands sooner
        if the player moves. Real ring requests always preempt via priority."""
        import math
        while True:
            time.sleep(4)
            if self.last_centre is None:
                continue
            with self.cv:
                idle = not self.heap
            if not idle:
                continue
            x_m, z_m = self.last_centre
            tc0, tr0 = T.tile_of_pos_m(x_m, z_m, self.spec.span)
            for ring_r in (3, 2):                    # outer first
                for dtr in range(-ring_r, ring_r + 1):
                    for dtc in range(-ring_r, ring_r + 1):
                        if max(abs(dtc), abs(dtr)) != ring_r:
                            continue                # only the ring shell
                        tc, tr = tc0 + dtc, tr0 + dtr
                        if self.cache.get((tc, tr)) is not None:
                            continue
                        d = math.hypot(dtc, dtr)
                        self.request(tc, tr, 100 + d)  # far below any live priority


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        """Never answer with nothing.

        An unhandled exception in the router closes the connection having sent zero
        bytes. A client cannot distinguish that from a dropped network, so it retries,
        crashes the handler again, and the loop is invisible on both ends. Any escape
        gets logged and turned into a 500 with a body.
        """
        try:
            return self._route()
        except (BrokenPipeError, ConnectionResetError):
            return                                   # client hung up; nothing to say
        except Exception:
            traceback.print_exc()
            try:
                return self._json(500, {"error": "handler failed", "path": self.path})
            except Exception:
                return

    def _route(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        S = self.server.streamer

        def num(k, d, cast=float):
            # float("nan"), float("inf") and float("1e400") all PARSE, so the two
            # exceptions caught here let a non-finite through to tile_of_pos_m, where
            # int(nan // 2) raises inside do_GET and the connection closes having sent
            # zero bytes -- not a 500, nothing at all, which a client cannot tell from
            # a network fault.
            try:
                v = cast(q.get(k, [d])[0])
            except (TypeError, ValueError, OverflowError):
                return d
            if isinstance(v, float) and not math.isfinite(v):
                return d
            return v

        def clamped(k, d, lo, hi, cast=int):
            return max(lo, min(hi, num(k, d, cast)))

        if u.path == "/health":
            n = S.stats["built"]
            return self._json(200, {
                "ok": True, "seed": S.seed, "spec": T.asdict(S.spec),
                "model_resident": RC._pipeline is not None,
                "tiles_built": n, "tiles_served": S.stats["served"],
                "mean_build_s": round(S.stats["build_s"] / n, 3) if n else None,
                "queue": len(S.pending), "cache": S.cache.stats(),
                "last_timing": S.stats["last"]})

        if u.path == "/spec":
            return self._json(200, {
                # Tiles are served immutable for a year, which is correct while the world
                # is fixed and a trap the moment it is not: relighting the labyrinth left
                # every browser holding the previous, darker tiles. The version covers
                # everything that changes what a tile looks like, so a client that
                # includes it in the URL cannot be served a stale world.
                "version": S.version,
                "seed": S.seed, "span": S.spec.span, "cell_m": CELL_M,
                "wall_h_m": WALL_H * CELL_M, "eye_h_m": EYE_H * CELL_M,
                "stride_floats": 14,
                "layout": ["x", "y", "z", "sx", "sy", "sz",
                           "qw", "qx", "qy", "qz", "r", "g", "b", "a"],
                "note": "splat means are tile-relative; add origin_m to place"})

        if u.path == "/ring":
            # r is CLAMPED, as /maze and /occupancy already clamp cols/rows. It was the
            # one size parameter that reached the streamer unbounded, and ring() enqueues
            # (2r+1)^2 builds that nothing can cancel: r=200 is 160,801 tiles, 83 h of
            # GPU; r=1000 is 4,004,001, a 331 MB response body, and ~3.2 GB of pending
            # and heap still held a minute after the request is answered. The client
            # already clamps to 0..4 (endless.js numParam), so this only ever refuses a
            # hand-made request -- which is exactly the one that can do the damage.
            return self._json(200, S.ring(num("x", 0.0), num("z", 0.0),
                                          clamped("r", 1, 0, 4)))

        if u.path == "/maze":
            c0, r0 = num("c0", 0, int), num("r0", 0, int)
            cols, rows = num("cols", 16, int), num("rows", 16, int)
            cols, rows = min(cols, 128), min(rows, 128)
            walls = [[S.maze.has_wall(c0 + c, r0 + r, d) for d in DIRS]
                     for r in range(rows) for c in range(cols)]
            t = S.maze.torches_in(c0, r0, cols, rows)
            return self._json(200, {
                "c0": c0, "r0": r0, "cols": cols, "rows": rows,
                "walls": walls, "order": ["N", "E", "S", "W"],
                "torches": [[round(p[0] * CELL_M, 3), round(p[1] * CELL_M, 3),
                             round(p[2] * CELL_M, 3)] for p in t]})

        if u.path == "/occupancy":
            # The server's own rasterisation, so the client's rebuilt copy can be diffed
            # against it instead of trusted. A one-index disagreement here is a wall the
            # player walks through, which looks like a reconstruction fault and is not.
            c0, r0 = num("c0", 0, int), num("r0", 0, int)
            cols, rows = min(num("cols", 16, int), 64), min(num("rows", 16, int), 64)
            g = S.maze.occupancy(c0, r0, cols, rows)
            return self._json(200, {"c0": c0, "r0": r0, "cols": cols, "rows": rows,
                                    "grid": g.astype(int).tolist()})

        if u.path == "/tile":
            tc, tr = num("tc", 0, int), num("tr", 0, int)
            got = S.cache.get((tc, tr))
            if got is None:
                S.request(tc, tr, 0.0)
                if num("wait", 1, int):
                    deadline = time.time() + num("timeout", 120.0)
                    while got is None and time.time() < deadline:
                        time.sleep(0.05)
                        got = S.cache.get((tc, tr))
                if got is None:
                    return self._json(202, {"state": "queued", "tile": [tc, tr]})
            payload, meta = got
            S.stats["served"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Tile-Meta", json.dumps(meta))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Expose-Headers", "X-Tile-Meta")
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.end_headers()
            return self.wfile.write(payload)

        self._json(404, {"error": "use /health /spec /ring /tile /maze"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--seed", type=int, default=1000)
    # Defaults come from TileSpec, never restated here. Restating them is how the server
    # ran at size=504 after TileSpec had been changed to 420, and how it crashed earlier
    # passing views= to a TileSpec that no longer had that field. One source of truth.
    _d = T.TileSpec()
    ap.add_argument("--span", type=int, default=_d.span)
    ap.add_argument("--n-pos", type=int, default=_d.n_pos, dest="n_pos")
    ap.add_argument("--n-az", type=int, default=_d.n_az, dest="n_az")
    ap.add_argument("--size", type=int, default=_d.size)
    ap.add_argument("--max-splats", type=int, default=_d.max_splats, dest="max_splats")
    ap.add_argument("--cache-gb", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=1,
                    help="tile builders; >1 measured slower, the GPU is the bottleneck")
    ap.add_argument("--warm", type=int, default=1,
                    help="tiles of radius to build before listening")
    a = ap.parse_args()

    spec = T.TileSpec(span=a.span, n_pos=a.n_pos, n_az=a.n_az, size=a.size,
                      max_splats=a.max_splats)
    S = Streamer(a.seed, spec, cache_gb=a.cache_gb, workers=a.workers)
    print(f"[server] seed={a.seed} spec={spec.key()}", flush=True)
    print(f"[server] tile {spec.span * 2 * CELL_M:.0f}x{spec.span * 2 * CELL_M:.0f} m, "
          f"{spec.n_pos}x{spec.n_az}={spec.views} views at {spec.size}px, "
          f"budget {spec.max_splats:,} splats", flush=True)
    print(f"[server] torch every {S.maze.spacing * CELL_M:.1f} m, "
          f"falloff k={R2.FALLOFF_K}, fill={R2.FILL}, reach={R2.REACH_M} m", flush=True)
    print(f"[server] world version {S.version}", flush=True)

    if a.warm:
        RC.get_pipeline()
        ox, oz = T.tile_origin_m(0, 0, spec.span)
        S.ring(ox, oz, radius=a.warm)
        while S.pending or S.cache.stats()["tiles"] == 0:
            time.sleep(0.2)
        print(f"[server] warm: {S.cache.stats()}", flush=True)

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.streamer = S
    print(f"[server] listening on http://{a.host}:{a.port}  "
          f"(/health, /ring?x=0&z=0&r=1)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
