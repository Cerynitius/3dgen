#!/usr/bin/env python
"""Infinite labyrinth: deterministic, locally computable, provably connected.

The old Cells class could not grow. Its walls array, the flood fill in _repair and
the global spacing sort in _torches each needed the whole grid resident before a
single cell was correct, so "endless" was never a parameter you could raise -- it
needed a different algorithm.

This module generates the same kind of labyrinth (corridors, rooms, doors, braided
dead ends, torch brackets) over the whole of Z^2, from local information only:

  chunk        K x K cells carved by a seeded backtracker -> internally connected
  border gate  the openings on a shared chunk edge are hash(seed, edge) alone, so
               two chunks derive an identical gate set without ever meeting
  torch        accepted where its hash is a local maximum within the spacing radius
               -- a Poisson-disk rule that needs only a bounded neighbourhood

Connectivity over the infinite plane then holds by construction rather than by
repair: every chunk is connected inside, and every pair of adjacent chunks shares
at least one gate, so the chunk adjacency graph is the full grid graph on Z^2.

Cell coordinates are global signed integers. Nothing here is sized by the world.
"""
import random
from collections import OrderedDict

import numpy as np

# ---------------------------------------------------------------- world constants
# Shared with the renderer and the metric alignment in tiles.py. A chunk of the
# reconstruction is only placeable on a global lattice because these are fixed.
CELL_M = 2.0                 # metres per grid unit
WALL_H = 3.2 / CELL_M        # ceiling height, in grid units
EYE_H = 1.65 / CELL_M        # eye height, in grid units

N, E, S, W = 0, 1, 2, 3
DC = [0, 1, 0, -1]
DR = [-1, 0, 1, 0]
OPP = [S, W, N, E]
DIRS = [N, E, S, W]

CHUNK = 8                    # cells per chunk side
MASK64 = (1 << 64) - 1

# hash domains, kept distinct so the same coordinates never collide across uses
_D_CHUNK = 0xC0FFEE
_D_GATE_V = 0x11
_D_GATE_H = 0x22
_D_TORCH = 0x33


def _hash(*vals):
    """splitmix64 over a tuple of ints. Negative coordinates fold in as two's
    complement, so the west half of the world is as well mixed as the east."""
    h = 0x9E3779B97F4A7C15
    for v in vals:
        h = (h ^ (int(v) & MASK64)) & MASK64
        h = (h * 0xBF58476D1CE4E5B9) & MASK64
        h ^= h >> 30
        h = (h * 0x94D049BB133111EB) & MASK64
        h ^= h >> 27
    h ^= h >> 31
    return h & MASK64


def _hash01(*vals):
    return _hash(*vals) / 18446744073709551616.0


def _fdiv(a, b):
    return a // b            # python floor division is already correct for negatives


# ---------------------------------------------------------------- one chunk
class Chunk:
    """K x K cells with their own wall bits. Generated from (seed, ch, rh) and the
    four hashed gate sets, so it is byte-identical whoever asks for it and in
    whatever order chunks happen to be visited."""

    __slots__ = ("ch", "rh", "k", "walls", "rooms", "room", "torches")

    def __init__(self, seed, ch, rh, k=CHUNK, braid=0.35, room_tries=6):
        self.ch, self.rh, self.k = ch, rh, k
        self.walls = [0b1111] * (k * k)
        self.room = [-1] * (k * k)
        self.rooms = []
        self.torches = None                      # filled lazily, needs a halo
        rnd = random.Random(_hash(seed, _D_CHUNK, ch, rh))
        self._carve_rooms(rnd, room_tries)
        self._grow(rnd)
        self._doors(rnd)
        self._braid(rnd, braid)
        self._repair()
        self._open_gates(seed)

    # -- local helpers, all in chunk-local cell coordinates [0,k)
    def _i(self, c, r):
        return r * self.k + c

    def _in(self, c, r):
        return 0 <= c < self.k and 0 <= r < self.k

    def _wall(self, c, r, d):
        return bool(self.walls[self._i(c, r)] & (1 << d))

    def _carve(self, c, r, d):
        nc, nr = c + DC[d], r + DR[d]
        if not self._in(nc, nr):
            return False
        self.walls[self._i(c, r)] &= ~(1 << d)
        self.walls[self._i(nc, nr)] &= ~(1 << OPP[d])
        return True

    def _exits(self, c, r):
        return [d for d in DIRS if not self._wall(c, r, d)]

    def _carve_rooms(self, rnd, tries):
        """Rooms stay a full cell clear of the chunk border, so a room never has to
        be agreed on by two chunks -- the one case the gate trick cannot cover."""
        for _ in range(tries):
            w, h = rnd.randint(2, 4), rnd.randint(2, 4)
            if self.k - w - 2 < 1 or self.k - h - 2 < 1:
                continue
            c0 = rnd.randint(1, self.k - w - 2)
            r0 = rnd.randint(1, self.k - h - 2)
            clash = any(self._in(c, r) and self.room[self._i(c, r)] >= 0
                        for r in range(r0 - 1, r0 + h + 1)
                        for c in range(c0 - 1, c0 + w + 1))
            if clash:
                continue
            rid = len(self.rooms)
            for r in range(r0, r0 + h):
                for c in range(c0, c0 + w):
                    self.room[self._i(c, r)] = rid
                    if c + 1 < c0 + w:
                        self._carve(c, r, E)
                    if r + 1 < r0 + h:
                        self._carve(c, r, S)
            self.rooms.append((c0, r0, w, h))

    def _grow(self, rnd):
        seen = [self.room[i] >= 0 for i in range(self.k * self.k)]

        def first_free():
            for r in range(self.k):
                for c in range(self.k):
                    if not seen[self._i(c, r)]:
                        return c, r
            return None

        start = first_free()
        while start is not None:
            c, r = start
            seen[self._i(c, r)] = True
            stack = [(c, r, None)]
            while stack:
                c, r, came = stack[-1]
                opts = [d for d in DIRS
                        if self._in(c + DC[d], r + DR[d])
                        and not seen[self._i(c + DC[d], r + DR[d])]]
                if not opts:
                    stack.pop()
                    continue
                # bias towards continuing straight so corridors read as corridors
                d = came if (came in opts and rnd.random() < 0.6) else rnd.choice(opts)
                self._carve(c, r, d)
                nc, nr = c + DC[d], r + DR[d]
                seen[self._i(nc, nr)] = True
                stack.append((nc, nr, d))
            start = first_free()

    def _doors(self, rnd):
        for rid, (c0, r0, w, h) in enumerate(self.rooms):
            cands = []
            for r in range(r0, r0 + h):
                for c in range(c0, c0 + w):
                    for d in DIRS:
                        nc, nr = c + DC[d], r + DR[d]
                        if self._in(nc, nr) and self.room[self._i(nc, nr)] != rid:
                            cands.append((c, r, d))
            for _ in range(rnd.randint(1, 3)):
                if not cands:
                    break
                c, r, d = rnd.choice(cands)
                self._carve(c, r, d)

    def _braid(self, rnd, braid):
        ends = [(c, r) for r in range(self.k) for c in range(self.k)
                if len(self._exits(c, r)) == 1]
        for c, r in ends:
            if rnd.random() > braid or len(self._exits(c, r)) != 1:
                continue
            shut = [d for d in DIRS
                    if self._wall(c, r, d) and self._in(c + DC[d], r + DR[d])]
            if shut:
                self._carve(c, r, rnd.choice(shut))

    def _flood(self, start):
        seen = {self._i(*start)}
        q = [start]
        while q:
            c, r = q.pop()
            for d in self._exits(c, r):
                nc, nr = c + DC[d], r + DR[d]
                if self._in(nc, nr) and self._i(nc, nr) not in seen:
                    seen.add(self._i(nc, nr))
                    q.append((nc, nr))
        return seen

    def _repair(self):
        """Guarantee the one property the plane-wide argument rests on: this chunk
        is a single connected component before any gate is opened."""
        for _ in range(self.k * self.k):
            main = self._flood((0, 0))
            if len(main) == self.k * self.k:
                return
            joined = False
            for r in range(self.k):
                for c in range(self.k):
                    if self._i(c, r) in main:
                        continue
                    for d in DIRS:
                        nc, nr = c + DC[d], r + DR[d]
                        if self._in(nc, nr) and self._i(nc, nr) in main:
                            self._carve(c, r, d)
                            joined = True
                            break
                    if joined:
                        break
                if joined:
                    break
            if not joined:
                return

    def _open_gates(self, seed):
        """Open this chunk's half of each shared border. The neighbour computes the
        same set from the same edge id, so the two halves always agree."""
        for row in gates_v(seed, self.ch, self.rh, self.k):          # east border
            self.walls[self._i(self.k - 1, row)] &= ~(1 << E)
        for row in gates_v(seed, self.ch - 1, self.rh, self.k):      # west border
            self.walls[self._i(0, row)] &= ~(1 << W)
        for col in gates_h(seed, self.ch, self.rh, self.k):          # south border
            self.walls[self._i(col, self.k - 1)] &= ~(1 << S)
        for col in gates_h(seed, self.ch, self.rh - 1, self.k):      # north border
            self.walls[self._i(col, 0)] &= ~(1 << N)


def gates_v(seed, ch, rh, k=CHUNK):
    """Open rows on the vertical border between chunk (ch,rh) and (ch+1,rh).

    Derived from the edge alone. At least one, so the plane cannot fall apart."""
    n = 1 + _hash(seed, _D_GATE_V, ch, rh) % 3
    order = sorted(range(k), key=lambda i: _hash(seed, _D_GATE_V + 1, ch, rh, i))
    return sorted(order[:n])


def gates_h(seed, ch, rh, k=CHUNK):
    """Open columns on the horizontal border between chunk (ch,rh) and (ch,rh+1)."""
    n = 1 + _hash(seed, _D_GATE_H, ch, rh) % 3
    order = sorted(range(k), key=lambda i: _hash(seed, _D_GATE_H + 1, ch, rh, i))
    return sorted(order[:n])


# ---------------------------------------------------------------- the world
class InfiniteMaze:
    """The labyrinth over all of Z^2, materialised one chunk at a time.

    Memory is bounded by the LRU, not by how far anyone has walked."""

    # spacing_cells 1.25 = a torch every 5 m. At the original 2.5 (10 m) the nearest
    # torch was 8 m away down a corridor and inverse-square had already taken it to 0.04,
    # leaving the world at a median pixel of 0.09 -- see bench_light.py, which sweeps this
    # against a stated brightness target rather than to taste.
    def __init__(self, seed=1000, k=CHUNK, cache=4096, spacing_cells=1.25):
        self.seed, self.k = seed, k
        self.spacing = spacing_cells * 2.0        # grid units; a cell spans two
        self._cache = OrderedDict()
        self._cap = cache
        self.built = 0

    # -- chunk access
    def chunk(self, ch, rh):
        key = (ch, rh)
        got = self._cache.get(key)
        if got is None:
            got = Chunk(self.seed, ch, rh, self.k)
            self.built += 1
            self._cache[key] = got
            if len(self._cache) > self._cap:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return got

    def chunk_of(self, c, r):
        return self.chunk(_fdiv(c, self.k), _fdiv(r, self.k))

    # -- cell queries, in global coordinates
    def has_wall(self, c, r, d):
        ck = self.chunk_of(c, r)
        return ck._wall(c - ck.ch * self.k, r - ck.rh * self.k, d)

    def exits(self, c, r):
        return [d for d in DIRS if not self.has_wall(c, r, d)]

    def in_room(self, c, r):
        ck = self.chunk_of(c, r)
        return ck.room[ck._i(c - ck.ch * self.k, r - ck.rh * self.k)] >= 0

    # -- rasterisation for the raycaster
    def occupancy(self, c0, r0, cols, rows):
        """The odd/even grid the renderer walks, for the cell box [c0,c0+cols) x
        [r0,r0+rows). True = solid.

        The outer ring is left solid on purpose: it is the window's skirt. A doorway
        written onto it would let a ray escape the array entirely and come back as a
        black hole, since a ray that hits nothing has no surface to shade."""
        occ = np.ones((2 * rows + 1, 2 * cols + 1), dtype=bool)
        for r in range(r0, r0 + rows):
            jr = 2 * (r - r0) + 1
            for c in range(c0, c0 + cols):
                jc = 2 * (c - c0) + 1
                occ[jr, jc] = False
                for d in DIRS:
                    if not self.has_wall(c, r, d):
                        nr_, nc_ = jr + DR[d], jc + DC[d]
                        if 0 < nr_ < 2 * rows and 0 < nc_ < 2 * cols:
                            occ[nr_, nc_] = False
        return occ

    # -- torches
    def _chunk_torches(self, ch, rh):
        """Torches whose cell lies in chunk (ch,rh), by the local-maximum rule.

        Acceptance depends only on candidates within the spacing radius, so it is
        the same whether this chunk is reached first or last -- and a torch never
        moves when the player walks back."""
        ck = self.chunk(ch, rh)
        if ck.torches is not None:
            return ck.torches
        halo = int(np.ceil(self.spacing / 2.0)) + 1
        c0, r0 = ch * self.k - halo, rh * self.k - halo
        cols = rows = self.k + 2 * halo
        pos, wgt, own = [], [], []
        for r in range(r0, r0 + rows):
            for c in range(c0, c0 + cols):
                inside = (ch * self.k <= c < (ch + 1) * self.k and
                          rh * self.k <= r < (rh + 1) * self.k)
                for d in DIRS:
                    if not self.has_wall(c, r, d):
                        continue
                    pos.append((2 * c + 1.5 + DC[d] * 0.38,
                                2 * r + 1.5 + DR[d] * 0.38))
                    wgt.append(_hash(self.seed, _D_TORCH, c, r, d))
                    own.append((inside, c, r, d))
        if not pos:
            ck.torches = []
            return ck.torches
        p = np.asarray(pos, dtype=np.float64)
        w = np.asarray(wgt, dtype=np.float64)
        out = []
        for i, (inside, c, r, d) in enumerate(own):
            if not inside:
                continue
            near = ((p[:, 0] - p[i, 0]) ** 2 + (p[:, 1] - p[i, 1]) ** 2) < self.spacing ** 2
            near[i] = False
            if near.any() and w[near].max() >= w[i]:
                continue                          # a stronger torch already claims it
            out.append((float(p[i, 0]), float(WALL_H * 0.55), float(p[i, 1]), c, r, d))
        ck.torches = out
        return out

    def torches_in(self, c0, r0, cols, rows):
        """Every torch whose cell lies in the box, gathered from the chunks it spans."""
        out = []
        for rh in range(_fdiv(r0, self.k), _fdiv(r0 + rows - 1, self.k) + 1):
            for ch in range(_fdiv(c0, self.k), _fdiv(c0 + cols - 1, self.k) + 1):
                for t in self._chunk_torches(ch, rh):
                    if c0 <= t[3] < c0 + cols and r0 <= t[4] < r0 + rows:
                        out.append(t)
        return out

    # -- navigation
    def bfs(self, start, goal, limit=200000):
        """Shortest cell path, used by the selftest and by trajectory planning.
        Explores the world as it goes -- no bounding box needed."""
        from collections import deque
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                path, node = [], cur
                while node is not None:
                    path.append(node)
                    node = prev[node]
                return path[::-1]
            if len(prev) > limit:
                return None
            c, r = cur
            for d in self.exits(c, r):
                nxt = (c + DC[d], r + DR[d])
                if nxt not in prev:
                    prev[nxt] = cur
                    q.append(nxt)
        return None


# ---------------------------------------------------------------- geometry helpers
def cell_centre_grid(c, r):
    """Cell centre in grid units. Open cell (c,r) occupies odd index (2c+1, 2r+1),
    whose centre is half a unit further on."""
    return 2 * c + 1.5, 2 * r + 1.5


def cell_centre_m(c, r):
    x, z = cell_centre_grid(c, r)
    return x * CELL_M, z * CELL_M
