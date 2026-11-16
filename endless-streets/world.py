"""Endless procedural town: an irregular street network (main avenues, streets and lanes, T-junctions where a street
segment is left out), perimeter blocks whose buildings have podiums, set-back towers, forecourts, gaps and roof
structures, residential compounds, schools, squares, market halls and parks, street furniture along every segment,
and landmarks, traffic lights and street-name signs at the crossings.

Everything is a pure function of the seed: a street line's position and class come from the line's own hash, a
segment's presence from the hashes of the segment and of its two crossings, and a block's contents from the block's
hash, so any window of the town can be built on its own and a place you walk back to is the place you left.

Frame: metres, Y up, ground at y = 0. Two families of street lines: family 0 are the lines z = Z(k), carrying the
streets along x; family 1 are the lines x = X(k), carrying the streets along z. Crossing (i, j) sits at (X(i), Z(j));
block (i, j) is the cell between x-lines i, i+1 and z-lines j, j+1. Segment (fam, k, m) is the stretch of line k of
a family between the crossings with lines m and m+1 of the other family.

Primitives are axis-aligned boxes and spheres, each with a material id and eight float parameters the shader reads,
plus props: objects whose look is a generated cut-out or photo (cityassets).
"""
import math
from functools import lru_cache

import numpy as np

P = 44.0            # mean spacing of parallel street lines
JIT = 7.0           # a line sits up to this far from its regular place: blocks are 30 .. 58 m across
GF = 4.2            # ground-floor height (shop front + sign band)
FH = 3.2            # upper floor height
EYE = 1.6           # eye height of a walker

# street classes: carriageway half width and corridor half width (carriageway + sidewalks)
LANE, STREET, MAIN = 0, 1, 2
ROAD_OF = (2.5, 3.5, 7.0)
HW_OF = (3.8, 6.0, 10.5)
ROAD, HW = ROAD_OF[STREET], HW_OF[STREET]      # an ordinary street
FLOORS_OF = ((1, 3), (2, 7), (5, 11))          # floors of a street-front building, by the class of its street

# materials
M_FACADE, M_ROOF, M_TRUNK, M_CROWN, M_POLE, M_LAMP, M_KIOSK, M_STONE, M_ADCOL, M_BENCH, M_BRONZE = range(1, 12)
M_GROUND = 0        # the y = 0 plane (asphalt / sidewalk / markings / block grounds), not a primitive
SKY = -1

# what a photo box is (stored as kind * 100000 + seed in box parameter 6; cityassets writes the prompt from it)
F_SHOP, F_UPPER, F_HOUSING, F_SCHOOL, F_MARKET, F_GATEHOUSE, F_ROOFHOUSE, F_TANK, F_WALL = range(9)
FKIND = 100000

# block kinds
B_PERIMETER, B_PARK, B_SQUARE, B_COMPOUND, B_SCHOOL, B_MARKET = range(6)

PALETTE_WALL = np.array([
    [0.78, 0.70, 0.58], [0.85, 0.80, 0.70], [0.62, 0.36, 0.28], [0.70, 0.45, 0.33],
    [0.80, 0.76, 0.66], [0.55, 0.58, 0.60], [0.86, 0.74, 0.52], [0.68, 0.72, 0.66],
    [0.80, 0.62, 0.55], [0.74, 0.74, 0.78], [0.58, 0.50, 0.44], [0.90, 0.86, 0.78]], np.float32)
PALETTE_SIGN = np.array([
    [0.75, 0.10, 0.10], [0.10, 0.30, 0.65], [0.10, 0.50, 0.25], [0.90, 0.70, 0.10],
    [0.15, 0.15, 0.15], [0.55, 0.15, 0.50], [0.95, 0.45, 0.10], [0.05, 0.55, 0.60],
    [0.85, 0.85, 0.82], [0.45, 0.25, 0.12]], np.float32)
PALETTE_BRIGHT = np.array([
    [0.80, 0.15, 0.12], [0.12, 0.35, 0.75], [0.15, 0.60, 0.30], [0.95, 0.75, 0.15],
    [0.60, 0.20, 0.60], [0.95, 0.50, 0.15]], np.float32)


def h32(*xs):
    """Deterministic 32-bit hash of integers (splitmix-style)."""
    v = 0x9E3779B9
    for x in xs:
        v = (v ^ (int(x) & 0xFFFFFFFF)) * 0x85EBCA6B & 0xFFFFFFFF
        v ^= v >> 13
        v = v * 0xC2B2AE35 & 0xFFFFFFFF
        v ^= v >> 16
    return v


class Rng:
    """A tiny counter-based generator so a block's draws do not depend on call order elsewhere."""
    def __init__(self, *key):
        self.key, self.n = key, 0

    def u(self):
        self.n += 1
        return h32(*self.key, self.n) / 4294967296.0

    def r(self, a, b):
        return a + (b - a) * self.u()

    def i(self, n):
        return int(self.u() * n) % n


# ------------------------------------------------------------------ street network
@lru_cache(maxsize=65536)
def line(fam, k, seed=0):
    """(position, class) of line k of a family. Every fourth line (from a seed offset) is a main avenue; of the
    others about a third are lanes. Line 0 of each family stays at 0 (the home street, routes.py)."""
    rng = Rng(seed, 201, fam, k)
    pos = 0.0 if k == 0 else k * P + rng.r(-JIT, JIT)
    if (k + h32(seed, 203, fam)) % 4 == 0:
        cls = MAIN
    elif rng.u() < 0.33:
        cls = LANE
    else:
        cls = STREET
    return pos, cls


def lpos(fam, k, seed=0):
    return line(fam, k, seed)[0]


def lcls(fam, k, seed=0):
    return line(fam, k, seed)[1]


def lroad(fam, k, seed=0):
    return ROAD_OF[line(fam, k, seed)[1]]


def lhw(fam, k, seed=0):
    return HW_OF[line(fam, k, seed)[1]]


def cell_of(fam, v, seed=0):
    """Index k of the line of a family at or below coordinate v (lines k and k+1 bound the cell holding v)."""
    k = math.floor(v / P)
    while lpos(fam, k, seed) > v:
        k -= 1
    while lpos(fam, k + 1, seed) <= v:
        k += 1
    return k


def _token(i, j, seed):
    """Which family may lose its one removable arm at crossing (i, j)."""
    return 0 if Rng(seed, 211, i, j).u() < 0.5 else 1


@lru_cache(maxsize=262144)
def seg(fam, k, m, seed=0):
    """Whether segment (fam, k, m) exists. Only one arm per family per crossing is removable (a parity rule), and a
    crossing's token lets only one family lose it, so every crossing keeps at least three arms (a T-junction at
    worst), every block misses at most one side, and the network stays connected. Main avenues are never cut."""
    if lcls(fam, k, seed) == MAIN or (m + k) % 2 != fam or (fam, k, m) == (1, 0, -1):    # the last: the home street
        return True
    a, b = ((m, k), (m + 1, k)) if fam == 0 else ((k, m), (k, m + 1))
    if _token(*a, seed) != fam or _token(*b, seed) != fam:
        return True
    return Rng(seed, 213, fam, k, m).u() >= 0.9


def arms(i, j, seed=0):
    """Arms of crossing (i, j) that exist: west, east (the street along x), south, north (the street along z)."""
    return dict(W=seg(0, j, i - 1, seed), E=seg(0, j, i, seed), S=seg(1, i, j - 1, seed), N=seg(1, i, j, seed))


def seg_geom(fam, k, m, seed=0):
    """Centre line c, along-range a0..a1 between the crossings' centres, and the along-range between the crossing
    boxes (where the street furniture stands)."""
    c = lpos(fam, k, seed)
    a0, a1 = lpos(1 - fam, m, seed), lpos(1 - fam, m + 1, seed)
    return c, a0, a1, a0 + lhw(1 - fam, m, seed), a1 - lhw(1 - fam, m + 1, seed)


def segments_in(xmin, xmax, zmin, zmax, seed=0, margin=0.0):
    """Existing segments whose corridor overlaps the rectangle: dicts fam, k, m, c, a0, a1, road, hw, cls."""
    out = []
    x0, x1, z0, z1 = xmin - margin, xmax + margin, zmin - margin, zmax + margin
    for fam, (lo, hi, alo, ahi) in ((0, (z0, z1, x0, x1)), (1, (x0, x1, z0, z1))):
        for k in range(cell_of(fam, lo - 11, seed), cell_of(fam, hi + 11, seed) + 2):
            c, hw = lpos(fam, k, seed), lhw(fam, k, seed)
            if c + hw < lo or c - hw > hi:
                continue
            for m in range(cell_of(1 - fam, alo, seed), cell_of(1 - fam, ahi, seed) + 1):
                if not seg(fam, k, m, seed):
                    continue
                c_, a0, a1, _, _ = seg_geom(fam, k, m, seed)
                if a1 < alo - 11 or a0 > ahi + 11:
                    continue
                out.append(dict(fam=fam, k=k, m=m, c=c, a0=a0, a1=a1, road=lroad(fam, k, seed), hw=hw,
                                cls=lcls(fam, k, seed)))
    return out


def crossings_in(xmin, xmax, zmin, zmax, seed=0):
    """Crossings inside the rectangle: (i, j, x, z)."""
    out = []
    for i in range(cell_of(1, xmin, seed), cell_of(1, xmax, seed) + 2):
        x = lpos(1, i, seed)
        if not xmin <= x < xmax:
            continue
        for j in range(cell_of(0, zmin, seed), cell_of(0, zmax, seed) + 2):
            z = lpos(0, j, seed)
            if zmin <= z < zmax:
                out.append((i, j, x, z))
    return out


def on_street(x, z, seed=0):
    """Whether a ground point lies on a street corridor (carriageway or sidewalk)."""
    i, j = cell_of(1, x, seed), cell_of(0, z, seed)
    for k in (i, i + 1):
        if abs(x - lpos(1, k, seed)) < lhw(1, k, seed) and seg(1, k, j, seed):
            return True
    for k in (j, j + 1):
        if abs(z - lpos(0, k, seed)) < lhw(0, k, seed) and seg(0, k, i, seed):
            return True
    return False


@lru_cache(maxsize=16384)
def block_bounds(i, j, seed=0):
    """Cell edges X0, X1, Z0, Z1, which of its sides S, N, W, E face a street, and the inner rectangle (inside the
    street corridors; a side without a street reaches the cell edge, where the neighbouring cell continues)."""
    X0, X1, Z0, Z1 = lpos(1, i, seed), lpos(1, i + 1, seed), lpos(0, j, seed), lpos(0, j + 1, seed)
    e = dict(S=seg(0, j, i, seed), N=seg(0, j + 1, i, seed), W=seg(1, i, j, seed), E=seg(1, i + 1, j, seed))
    bx0 = X0 + (lhw(1, i, seed) if e["W"] else 0.0)
    bx1 = X1 - (lhw(1, i + 1, seed) if e["E"] else 0.0)
    bz0 = Z0 + (lhw(0, j, seed) if e["S"] else 0.0)
    bz1 = Z1 - (lhw(0, j + 1, seed) if e["N"] else 0.0)
    return X0, X1, Z0, Z1, e, (bx0, bx1, bz0, bz1)


@lru_cache(maxsize=16384)
def block_kind(i, j, seed=0):
    if (i, j) in ((-1, -1), (0, -1)):          # the two blocks on the home street (routes.home)
        return B_PERIMETER
    bx0, bx1, bz0, bz1 = block_bounds(i, j, seed)[5]
    w, d = bx1 - bx0, bz1 - bz0
    u = Rng(seed, 221, i, j).u()
    if u < 0.07:
        return B_PARK
    if u < 0.12:
        return B_SQUARE
    if u < 0.28:
        return B_COMPOUND if w >= 26 and d >= 26 else B_PERIMETER
    if u < 0.33:
        return B_SCHOOL if w >= 30 and d >= 28 else B_PERIMETER
    if u < 0.37:
        return B_MARKET if min(w, d) >= 18 else B_PERIMETER
    return B_PERIMETER


def front_of(x0, z0, x1, z1, seed=0):
    """The street a building box fronts: (axis, side, fam, k) -- axis 0: the front faces +-x (a street along z),
    2: +-z; side: +1 / -1, the sign of the front's outward normal -- or None (not on a street: compound slabs,
    set-back towers). A face fronts a street when its plane lies at the corridor edge or up to a forecourt behind
    it, on the street's side, along an existing segment."""
    best = None
    for fam, (p0, p1), (a0, a1) in ((0, (z0, z1), (x0, x1)), (1, (x0, x1), (z0, z1))):
        mid = (p0 + p1) / 2
        for side, pl in ((-1, p0), (1, p1)):
            k0 = cell_of(fam, pl, seed)
            for k in (k0 - 1, k0, k0 + 1, k0 + 2):
                c, hw = lpos(fam, k, seed), lhw(fam, k, seed)
                if (c - mid) * side <= 0:
                    continue                      # the line must lie in front of this face
                gap = abs(c - pl) - hw
                if not -0.3 < gap < 4.5:
                    continue
                am = (a0 + a1) / 2
                m = cell_of(1 - fam, am, seed)
                if not seg(fam, k, m, seed):
                    continue
                if best is None or gap < best[0]:
                    best = (gap, 2 if fam == 0 else 0, side, fam, k)
    return None if best is None else best[1:]


# ------------------------------------------------------------------ primitives
# Props: things whose look is generated while walking (cityassets). Columns of the props array:
# kind, x, y0, z, height, half width, angle (rad, of the billboard plane), crossed (0/1), group a, b, c
(P_STREET_TREE, P_PARK_TREE, P_BIG_TREE, P_STATUE, P_BENCH, P_KIOSK, P_ADCOL, P_CLOCK, P_LAMP,
 P_BUSSTOP, P_TLIGHT, P_SIGN, P_BIN, P_BIKES, P_PLANTER, P_SHELTER_SIDE) = range(1, 17)
BILLBOARD_PROPS = (P_STREET_TREE, P_PARK_TREE, P_BIG_TREE, P_STATUE, P_BENCH, P_LAMP,
                   P_BUSSTOP, P_TLIGHT, P_SIGN, P_BIN, P_BIKES, P_PLANTER, P_SHELTER_SIDE)   # generated cut-outs
WIDTH_PROPS = (P_BENCH, P_BUSSTOP, P_BIN, P_BIKES, P_PLANTER, P_SHELTER_SIDE)   # width fixed, height from the photo
NPROP = 11


class Prims:
    def __init__(self):
        self.box, self.boxp, self.sph, self.sphp = [], [], [], []
        self.boxk, self.sphk, self.props = [], [], []
        self.cur = -1                   # prop that the primitives being added belong to

    def add_box(self, lo, hi, mat, params):
        self.box.append([*lo, *hi, mat])
        self.boxp.append(list(params) + [0.0] * (8 - len(params)))
        self.boxk.append(self.cur)

    def add_sphere(self, c, r, mat, params):
        self.sph.append([*c, r, mat])
        self.sphp.append(list(params) + [0.0] * (8 - len(params)))
        self.sphk.append(self.cur)

    def prop(self, kind, x, y0, z, h, hw, ang=0.0, crossed=0, ga=0, gb=0, gc=0):
        self.props.append([kind, x, y0, z, h, hw, ang, crossed, ga, gb, gc])
        self.cur = len(self.props) - 1

    def arrays(self):
        f = lambda a, n: np.asarray(a, np.float32).reshape(-1, n)
        g = lambda a: np.asarray(a, np.int32).reshape(-1)
        return (f(self.box, 7), f(self.boxp, 8), f(self.sph, 5), f(self.sphp, 8),
                g(self.boxk), g(self.sphk), f(self.props, NPROP))


def _photo_box(pr, rng, lo, hi, kind, seed, wall=None):
    """A box whose vertical faces show a generated photo (cityassets.facade); the parameters keep the procedural
    look the shader falls back to while the photo is being generated."""
    wall = PALETTE_WALL[rng.i(len(PALETTE_WALL))] * rng.r(0.9, 1.06) if wall is None else wall
    ws, style, bay = rng.r(2.3, 3.6), rng.i(3), rng.r(2.8, 4.5)
    pr.add_box(lo, hi, M_FACADE, [*wall, ws, style + 4 * rng.i(len(PALETTE_SIGN)), bay,
                                  float(seed % FKIND + kind * FKIND), hi[1]])


def _roof_things(pr, rng, x0, z0, x1, z1, top, seed):
    """A stair / lift housing and sometimes a water tank on a flat roof."""
    w, d = x1 - x0, z1 - z0
    if w > 6 and d > 6 and rng.u() < 0.35:
        sx, sz = (3.0, 2.6) if rng.u() < 0.5 else (2.6, 3.0)
        cx, cz = rng.r(x0 + 1 + sx / 2, x1 - 1 - sx / 2), rng.r(z0 + 1 + sz / 2, z1 - 1 - sz / 2)
        _photo_box(pr, rng, (cx - sx / 2, top, cz - sz / 2), (cx + sx / 2, top + 2.8, cz + sz / 2), F_ROOFHOUSE,
                   h32(seed, 7), wall=np.array([0.70, 0.70, 0.68], np.float32))
    if w > 5 and d > 5 and rng.u() < 0.15:
        cx, cz = rng.r(x0 + 1.5, x1 - 1.5), rng.r(z0 + 1.5, z1 - 1.5)
        _photo_box(pr, rng, (cx - 1.1, top, cz - 0.8), (cx + 1.1, top + 1.9, cz + 0.8), F_TANK, h32(seed, 8),
                   wall=np.array([0.78, 0.80, 0.82], np.float32))


def _building(pr, rng, x0, z0, x1, z1, front, cls, corner, seed):
    """A street-front building on a lot; front = (dx, dz), the outward direction toward its street. Shapes: one box,
    a shop podium with a set-back tower, or a forecourt in front; corner lots stand taller; roofs carry a stair
    housing or a water tank."""
    lo, hi = FLOORS_OF[cls]
    floors = lo + rng.i(hi - lo + 1) + (2 if corner and cls != LANE else 0)
    depth = (z1 - z0) if front[1] else (x1 - x0)
    width = (x1 - x0) if front[1] else (z1 - z0)

    def shrink_front(x0, z0, x1, z1, s):
        if front == (0, -1):
            return x0, z0 + s, x1, z1
        if front == (0, 1):
            return x0, z0, x1, z1 - s
        if front == (-1, 0):
            return x0 + s, z0, x1, z1
        return x0, z0, x1 - s, z1

    if depth > 9 and rng.u() < 0.15:                       # forecourt: the building stands back from the sidewalk
        x0, z0, x1, z1 = shrink_front(x0, z0, x1, z1, rng.r(1.5, 3.0))
        depth = (z1 - z0) if front[1] else (x1 - x0)
    if floors >= 4 and depth > 8 and rng.u() < 0.45:      # podium + set-back tower
        pod = 1 + (rng.u() < 0.4)
        hp = GF + (pod - 1) * FH
        _photo_box(pr, rng, (x0, 0.0, z0), (x1, hp, z1), F_SHOP, seed)
        tx0, tz0, tx1, tz1 = shrink_front(x0, z0, x1, z1, min(rng.r(2.0, 4.5), depth - 5))
        if width > 9:
            s = rng.r(0.0, 1.2)
            if front[1]:
                tx0, tx1 = tx0 + s, tx1 - s
            else:
                tz0, tz1 = tz0 + s, tz1 - s
        top = hp + (floors - pod) * FH + rng.r(0.0, 0.6)
        _photo_box(pr, rng, (tx0, hp, tz0), (tx1, top, tz1), F_UPPER, seed)
        _roof_things(pr, rng, tx0, tz0, tx1, tz1, top, seed)
        return top
    top = GF + (floors - 1) * FH + rng.r(0.0, 0.6)
    _photo_box(pr, rng, (x0, 0.0, z0), (x1, top, z1), F_SHOP, seed)
    _roof_things(pr, rng, x0, z0, x1, z1, top, seed)
    return top


def _lots(rng, u0, u1, gap_p=0.12):
    """Split a frontage u0..u1 into lots 5.5 .. 11 m wide, with an occasional 2 .. 3.5 m gap (a passage into the
    block) between two lots."""
    out, u = [], u0
    while u < u1 - 0.5:
        if out and u1 - u > 14 and rng.u() < gap_p:
            u += rng.r(2.0, 3.5)
            continue
        w = min(rng.r(5.5, 11.0), u1 - u)
        if u1 - u - w < 4.0:
            w = u1 - u
        out.append((u, u + w))
        u += w
    return out


def _perimeter(pr, i, j, seed, e, b):
    """Rows of street-front buildings along every side of the block that faces a street."""
    rng = Rng(seed, i, j)
    bx0, bx1, bz0, bz1 = b
    w, d = bx1 - bx0, bz1 - bz0
    cls = dict(S=lcls(0, j, seed), N=lcls(0, j + 1, seed), W=lcls(1, i, seed), E=lcls(1, i + 1, seed))
    dep = {s: rng.r(9, 13) for s in "SNWE"}
    if e["S"] and e["N"]:
        dep["S"], dep["N"] = min(dep["S"], d / 2 - 0.25), min(dep["N"], d / 2 - 0.25)
    else:
        dep["S"], dep["N"] = min(dep["S"], d - 0.5), min(dep["N"], d - 0.5)
    if e["W"] and e["E"]:
        dep["W"], dep["E"] = min(dep["W"], w / 2 - 0.25), min(dep["E"], w / 2 - 0.25)
    else:
        dep["W"], dep["E"] = min(dep["W"], w - 0.5), min(dep["E"], w - 0.5)
    k = 0
    for s in "SN":
        if not e[s]:
            continue
        lots = _lots(rng, bx0, bx1)
        for n, (u0, u1) in enumerate(lots):
            z0, z1 = (bz0, bz0 + dep["S"]) if s == "S" else (bz1 - dep["N"], bz1)
            corner = (n == 0 and e["W"]) or (n == len(lots) - 1 and e["E"])
            _building(pr, rng, u0, z0, u1, z1, (0, -1) if s == "S" else (0, 1), cls[s], corner, h32(seed, i, j, k, 1))
            k += 1
    za = bz0 + (dep["S"] if e["S"] else 0.0)
    zb = bz1 - (dep["N"] if e["N"] else 0.0)
    for s in "WE":
        if not e[s] or zb - za < 4:
            continue
        for u0, u1 in _lots(rng, za, zb):
            x0, x1 = (bx0, bx0 + dep["W"]) if s == "W" else (bx1 - dep["E"], bx1)
            _building(pr, rng, x0, u0, x1, u1, (-1, 0) if s == "W" else (1, 0), cls[s], False, h32(seed, i, j, k, 2))
            k += 1


def _walls_and_gate(pr, rng, i, j, seed, e, b, inset=0.4):
    """A 2.2 m wall along every street side of a compound / school, in 6 m panels sharing one generated photo, with a
    gate (two pillars and a gatehouse) in the middle of the longest side. -> (gate side, gate centre)."""
    bx0, bx1, bz0, bz1 = b
    sides = [s for s in "SNWE" if e[s]]
    gs = max(sides, key=lambda s: (bx1 - bx0) if s in "SN" else (bz1 - bz0))
    wseed = h32(seed, i, j, 5)
    gate_c = None
    for s in sides:
        if s in "SN":
            u0, u1, c = bx0, bx1, (bz0 + inset if s == "S" else bz1 - inset)
        else:
            u0, u1, c = bz0, bz1, (bx0 + inset if s == "W" else bx1 - inset)
        gap = None
        if s == gs:
            gm = (u0 + u1) / 2 + rng.r(-0.15, 0.15) * (u1 - u0)
            gap, gate_c = (gm - 3.5, gm + 3.5), gm
        segs = [(u0, gap[0]), (gap[1], u1)] if gap else [(u0, u1)]
        for a, bb in segs:
            n = max(1, round((bb - a) / 6.0))
            for q in range(n):
                p0, p1 = a + (bb - a) * q / n, a + (bb - a) * (q + 1) / n
                lo, hi = ((p0, 0.0, c - 0.15), (p1, 2.2, c + 0.15)) if s in "SN" else ((c - 0.15, 0.0, p0), (c + 0.15, 2.2, p1))
                _photo_box(pr, rng, lo, hi, F_WALL, wseed, wall=np.array([0.80, 0.78, 0.74], np.float32))
        if gap:
            inward = 1 if s in "SW" else -1
            for g in gap:      # pillars
                lo, hi = ((g - 0.4, 0.0, c - 0.4), (g + 0.4, 3.0, c + 0.4)) if s in "SN" else ((c - 0.4, 0.0, g - 0.4), (c + 0.4, 3.0, g + 0.4))
                pr.add_box(lo, hi, M_STONE, [0.72, 0.70, 0.66])
            g0 = gap[1] + 0.6           # gatehouse just inside, beside the gate
            if s in "SN":
                lo, hi = (g0, 0.0, min(c, c + inward * 3.4)), (g0 + 3.6, 3.2, max(c, c + inward * 3.4))
            else:
                lo, hi = (min(c, c + inward * 3.4), 0.0, g0), (max(c, c + inward * 3.4), 3.2, g0 + 3.6)
            _photo_box(pr, rng, lo, hi, F_GATEHOUSE, h32(seed, i, j, 6))
    return gs, gate_c


def _compound(pr, i, j, seed, e, b):
    """A residential compound: walls and a gate, rows of slab blocks facing south, trees between the rows."""
    rng = Rng(seed, i, j, 31)
    bx0, bx1, bz0, bz1 = b
    _walls_and_gate(pr, rng, i, j, seed, e, b)
    floors = 6 + rng.i(6)
    z = bz0 + rng.r(7.0, 9.0)
    row = 0
    while z + 11 <= bz1 - 6:
        x = bx0 + 6
        span = bx1 - 6 - x
        n = 1 if span < 40 else 2
        for q in range(n):
            a = x + span * q / n + (1.5 if q else 0.0)
            bb = x + span * (q + 1) / n - (1.5 if q < n - 1 else 0.0)
            # long slabs are built from 12-14 m sections, each with its own photo
            m = max(1, round((bb - a) / 13.0))
            for s in range(m):
                p0, p1 = a + (bb - a) * s / m, a + (bb - a) * (s + 1) / m
                top = GF + (floors - 1) * FH + (0.0 if s else 0.3)
                _photo_box(pr, rng, (p0, 0.0, z), (p1, top, z + 11), F_HOUSING, h32(seed, i, j, row, q))
            _roof_things(pr, rng, a, z, bb, z + 11, GF + (floors - 1) * FH, h32(seed, i, j, row, q))
        zt = z + 11 + 4.0
        t = bx0 + 8
        while t < bx1 - 8 and zt < bz1 - 4:
            if rng.u() < 0.8:
                _tree(pr, t + rng.r(-1, 1), zt + rng.r(-0.5, 0.5), rng, big=rng.u() < 0.3, kind=P_PARK_TREE,
                      group=(i, j, 8))
            t += rng.r(7, 10)
        z += 11 + rng.r(15, 19)
        row += 1


def _school(pr, i, j, seed, e, b):
    """A school: walls and a gate, a teaching building along the side opposite the gate, a sports ground."""
    rng = Rng(seed, i, j, 41)
    bx0, bx1, bz0, bz1 = b
    gs, _ = _walls_and_gate(pr, rng, i, j, seed, e, b)
    floors = 4 + rng.i(2)
    top = GF + (floors - 1) * FH
    opp = {"S": "N", "N": "S", "W": "E", "E": "W"}[gs]
    if opp in "SN":
        a, bb = bx0 + 5, bx1 - 5
        z0, z1 = (bz1 - 4 - 14, bz1 - 4) if opp == "N" else (bz0 + 4, bz0 + 18)
        m = max(1, round((bb - a) / 14.0))
        for s in range(m):
            p0, p1 = a + (bb - a) * s / m, a + (bb - a) * (s + 1) / m
            _photo_box(pr, rng, (p0, 0.0, z0), (p1, top, z1), F_SCHOOL, h32(seed, i, j, 9))
    else:
        a, bb = bz0 + 5, bz1 - 5
        x0, x1 = (bx1 - 4 - 14, bx1 - 4) if opp == "E" else (bx0 + 4, bx0 + 18)
        m = max(1, round((bb - a) / 14.0))
        for s in range(m):
            p0, p1 = a + (bb - a) * s / m, a + (bb - a) * (s + 1) / m
            _photo_box(pr, rng, (x0, 0.0, p0), (x1, top, p1), F_SCHOOL, h32(seed, i, j, 9))


def _market(pr, i, j, seed, e, b):
    """A market hall: a low, wide hall in sections (each its own photo), shared bikes by the entrance."""
    rng = Rng(seed, i, j, 51)
    bx0, bx1, bz0, bz1 = b
    x0, x1, z0, z1 = bx0 + 2.5, bx1 - 2.5, bz0 + 2.5, bz1 - 2.5
    top = rng.r(7.0, 8.5)
    alongx = (x1 - x0) >= (z1 - z0)
    a, bb = (x0, x1) if alongx else (z0, z1)
    m = max(1, round((bb - a) / 12.0))
    for s in range(m):
        p0, p1 = a + (bb - a) * s / m, a + (bb - a) * (s + 1) / m
        lo, hi = ((p0, 0.0, z0), (p1, top, z1)) if alongx else ((x0, 0.0, p0), (x1, top, p1))
        _photo_box(pr, rng, lo, hi, F_MARKET, (h32(seed, i, j, 10) % 1000) * 10 + min(s, 9))   # section 0 has the sign


def _park(pr, i, j, seed, e, b):
    rng = Rng(seed, i, j)
    bx0, bx1, bz0, bz1 = b
    n = max(5, int((bx1 - bx0) * (bz1 - bz0) / 140))
    for k in range(n):
        _tree(pr, rng.r(bx0 + 2, bx1 - 2), rng.r(bz0 + 2, bz1 - 2), rng, big=rng.u() < 0.5, kind=P_PARK_TREE,
              group=(i, j, 8))
    for k in range(2):
        x, z = rng.r(bx0 + 4, bx1 - 4), rng.r(bz0 + 4, bz1 - 4)
        _bench(pr, x, z, 0.0, (i, j, 10 + k))


def _square(pr, i, j, seed, e, b):
    """A paved square: a statue on a pedestal in the middle, trees round the edge, benches facing the statue."""
    rng = Rng(seed, i, j, 61)
    bx0, bx1, bz0, bz1 = b
    cx, cz = (bx0 + bx1) / 2, (bz0 + bz1) / 2
    pr.add_box((cx - 2.0, 0.0, cz - 2.0), (cx + 2.0, 0.4, cz + 2.0), M_STONE, [0.62, 0.60, 0.56])
    pr.add_box((cx - 0.8, 0.4, cz - 0.8), (cx + 0.8, 2.0, cz + 0.8), M_STONE, [0.70, 0.68, 0.62])
    pr.prop(P_STATUE, cx, 2.0, cz, 3.3, 0.0, 0.0, 1, i, j, 77)
    pr.add_box((cx - 0.35, 2.0, cz - 0.35), (cx + 0.35, 4.4, cz + 0.35), M_BRONZE, [0.30, 0.42, 0.36])
    pr.cur = -1
    for (x, z) in ((cx, bz0 + 3), (cx, bz1 - 3), (bx0 + 3, cz), (bx1 - 3, cz)):
        pass
    t = bx0 + 3
    while t < bx1 - 3:
        for zz in (bz0 + 2.0, bz1 - 2.0):
            if rng.u() < 0.85:
                _tree(pr, t, zz, rng, kind=P_PARK_TREE, group=(i, j, 8))
        t += rng.r(7, 9)
    for k, (dx, dz, ang) in enumerate(((0, -6, 0.0), (0, 6, 0.0), (-6, 0, math.pi / 2), (6, 0, math.pi / 2))):
        _bench(pr, cx + dx, cz + dz, ang, (i, j, 20 + k))


def _bench(pr, x, z, ang, group):
    pr.prop(P_BENCH, x, 0.0, z, 0.0, 0.9, ang, 0, *group)
    if abs(math.sin(ang)) < 0.5:
        pr.add_box((x - 0.9, 0.0, z - 0.25), (x + 0.9, 0.45, z + 0.25), M_BENCH, [0.45, 0.30, 0.18])
    else:
        pr.add_box((x - 0.25, 0.0, z - 0.9), (x + 0.25, 0.45, z + 0.9), M_BENCH, [0.45, 0.30, 0.18])
    pr.cur = -1


def _tree(pr, x, z, rng, big=False, kind=P_STREET_TREE, group=(0, 0, 0)):
    s = 1.6 if big else 1.0
    pr.prop(kind, x, 0.0, z, 0.0, 0.0, 0.0, 1, *group)     # size and angle are set by cityassets
    pr.add_box((x - 0.14 * s, 0.0, z - 0.14 * s), (x + 0.14 * s, 2.6 * s, z + 0.14 * s), M_TRUNK, [0.36, 0.27, 0.20])
    g = np.array([0.22, 0.42, 0.18], np.float32) * rng.r(0.8, 1.15)
    pr.add_sphere((x, 3.6 * s, z), 1.6 * s, M_CROWN, [*g, rng.u() * 1000])     # stand-in until the cut-out exists
    pr.cur = -1


def _lamp(pr, x, z, alongx, sgn, group=(0, 0, 0)):
    """Street lamp: a generated cut-out (one design per street segment, like the trees); the pole box stands in
    until the image exists and stays for collisions."""
    pr.prop(P_LAMP, x, 0.0, z, 0.0, 0.0, 0.0, 1, *group)
    pr.add_box((x - 0.07, 0.0, z - 0.07), (x + 0.07, 5.2, z + 0.07), M_POLE, [0.20, 0.22, 0.22])
    pr.cur = -1


def _flat(pr, kind, x, z, hw, alongx, group, h_box, col):
    """A width-fixed cut-out standing parallel to the street (bus shelter, bins, bikes, planter) with a stand-in box."""
    pr.prop(kind, x, 0.0, z, 0.0, hw, 0.0 if alongx else math.pi / 2, 0, *group)
    t = 0.3
    lo, hi = ((x - hw, 0.0, z - t), (x + hw, h_box, z + t)) if alongx else ((x - t, 0.0, z - hw), (x + t, h_box, z + hw))
    pr.add_box(lo, hi, M_BENCH, col)
    pr.cur = -1


def _furniture(pr, fam, k, m, seed):
    """Street furniture along segment (fam, k, m): lamps at a steady spacing (both sides of an avenue, alternating
    sides of a street, one side of a lane), street trees between them (not in lanes), bins by some lamps, a bus
    shelter on some streets, shared bikes by the shop fronts, planters on avenues, benches by the shop fronts."""
    if not seg(fam, k, m, seed):
        return
    c, _, _, a0, a1 = seg_geom(fam, k, m, seed)
    a0, a1 = a0 + 1.0, a1 - 1.0
    L = a1 - a0
    if L < 6:
        return
    cls = lcls(fam, k, seed)
    road, hw = ROAD_OF[cls], HW_OF[cls]
    rng = Rng(seed, 231, fam, k, m)
    alongx = fam == 0
    grp = (fam, k, m)

    def at(t, off):
        return (t, c + off) if alongx else (c + off, t)
    curb = road + (0.5 if cls == LANE else 0.8)
    shop = hw - 0.6
    sides = (-1, 1)
    busy = {s: [] for s in sides}           # along-ranges already taken on each side
    # bus shelter
    if cls != LANE and L >= 26 and rng.u() < 0.45:
        s = sides[rng.i(2)]
        t = (a0 + a1) / 2 + rng.r(-0.15, 0.15) * L
        grp_b = (fam, k, m * 4 + (s > 0))
        _flat(pr, P_BUSSTOP, *at(t, s * (curb + 0.4)), 2.25, alongx, grp_b, 2.6, [0.55, 0.60, 0.62])
        for e in (-1, 1):       # side panels at both ends, across the street direction (seen from the sidewalk)
            _flat(pr, P_SHELTER_SIDE, *at(t + e * 2.2, s * (curb + 0.4)), 0.7, not alongx, grp_b, 2.2, [0.55, 0.60, 0.62])
        busy[s].append((t - 3.5, t + 3.5))
    # lamps
    sp = 20.0 if cls == MAIN else (24.0 if cls == STREET else 22.0)
    lamps = {s: [] for s in sides}
    for s in sides:
        if cls == LANE and s < 0:
            continue
        off = 0.0 if (cls == MAIN or s > 0) else sp / 2
        t = a0 + 2.0 + off
        while t < a1 - 2.0:
            if not any(lo <= t <= hi for lo, hi in busy[s]):
                _lamp(pr, *at(t, s * curb), alongx, -s, group=grp)
                lamps[s].append(t)
                busy[s].append((t - 1.5, t + 1.5))
            t += sp
    # bins beside some lamps
    for s in sides:
        for t in lamps[s]:
            if rng.u() < 0.45 and t + 1.6 < a1:
                _flat(pr, P_BIN, *at(t + 1.4, s * curb), 0.55, alongx, grp, 1.0, [0.25, 0.45, 0.30])
                busy[s].append((t + 0.8, t + 2.0))
    # street trees
    if cls != LANE:
        for s in sides:
            t = a0 + rng.r(1.5, 4.0)
            while t < a1 - 1.0:
                if not any(lo - 1.0 <= t <= hi + 1.0 for lo, hi in busy[s]) and rng.u() < 0.85:
                    _tree(pr, *at(t, s * curb), rng, group=grp)
                    busy[s].append((t - 1.0, t + 1.0))
                elif cls == MAIN and rng.u() < 0.35 and not any(lo - 1.6 <= t <= hi + 1.6 for lo, hi in busy[s]):
                    _flat(pr, P_PLANTER, *at(t, s * curb), 1.2, alongx, grp, 0.5, [0.60, 0.58, 0.55])
                    busy[s].append((t - 1.3, t + 1.3))
                t += rng.r(7.0, 9.0)
    # shared bikes by the shop fronts
    for s in sides:
        if rng.u() < (0.45 if cls != MAIN else 0.3) and L > 14:
            t = rng.r(a0 + 3, a1 - 9)
            for q in range(1 + rng.i(2)):
                _flat(pr, P_BIKES, *at(t + 2.3 * q + 1.1, s * shop), 1.1, alongx, grp, 0.9, [0.85, 0.55, 0.15])
    # benches by the shop fronts
    if cls != LANE and rng.u() < 0.5:
        s = sides[rng.i(2)]
        t = rng.r(a0 + 3, a1 - 3)
        _bench(pr, *at(t, s * (hw - 0.45)), 0.0 if alongx else math.pi / 2, (fam, k, m * 4 + 2 + (s > 0)))


def _landmark(pr, i, j, seed):
    """Landmarks at some crossings: what a route is remembered by. -> kind or None."""
    rng = Rng(seed, 77, i, j)
    if rng.u() > 0.55:
        return None
    c0, c1 = lcls(0, j, seed), lcls(1, i, seed)
    if c0 == LANE and c1 == LANE:
        return None
    cx, cz = lpos(1, i, seed), lpos(0, j, seed)
    kind = rng.i(5)
    col = PALETTE_BRIGHT[rng.i(len(PALETTE_BRIGHT))]
    corner = rng.i(4)
    sx, sz = (1 if corner in (0, 3) else -1), (1 if corner in (0, 1) else -1)
    # on the sidewalk corner: inside the wider street's sidewalk, clear of the other carriageway
    kx = cx + sx * max(HW_OF[c1] - 1.3, ROAD_OF[c1] + 1.2)
    kz = cz + sz * max(HW_OF[c0] - 1.3, ROAD_OF[c0] + 1.2)
    a = arms(i, j, seed)
    if kind == 2 and (LANE in (c0, c1) or not all(a.values())):
        kind = 3
    if kind == 0:      # kiosk with a coloured roof
        pr.prop(P_KIOSK, kx, 0.0, kz, 2.5, 1.0, ga=i, gb=j)
        pr.add_box((kx - 1.0, 0.0, kz - 1.0), (kx + 1.0, 2.5, kz + 1.0), M_KIOSK, [*col, 0.0, rng.u() * 1000])
        pr.cur = -1
        pr.add_box((kx - 1.25, 2.5, kz - 1.25), (kx + 1.25, 2.75, kz + 1.25), M_KIOSK, [*(col * 0.7), 1.0])
    elif kind == 1:    # advertising column
        pr.prop(P_ADCOL, kx, 0.0, kz, 3.2, 0.55, ga=i, gb=j)
        pr.add_box((kx - 0.55, 0.0, kz - 0.55), (kx + 0.55, 3.2, kz + 0.55), M_ADCOL, [*col, rng.u() * 1000])
        pr.cur = -1
        pr.add_box((kx - 0.65, 3.2, kz - 0.65), (kx + 0.65, 3.45, kz + 0.65), M_POLE, [0.15, 0.30, 0.20])
    elif kind == 2:    # statue on a roundabout island
        pr.add_box((cx - 2.2, 0.0, cz - 2.2), (cx + 2.2, 0.35, cz + 2.2), M_STONE, [0.62, 0.60, 0.56])
        pr.add_box((cx - 0.8, 0.35, cz - 0.8), (cx + 0.8, 2.2, cz + 0.8), M_STONE, [0.70, 0.68, 0.62])
        pr.prop(P_STATUE, cx, 2.2, cz, 3.3, 0.0, 0.0, 1, i, j)
        pr.add_box((cx - 0.35, 2.2, cz - 0.35), (cx + 0.35, 4.6, cz + 0.35), M_BRONZE, [0.30, 0.42, 0.36])
        pr.add_sphere((cx, 5.0, cz), 0.5, M_BRONZE, [0.30, 0.42, 0.36])
        pr.cur = -1
    elif kind == 3:    # clock pole
        pr.add_box((kx - 0.12, 0.0, kz - 0.12), (kx + 0.12, 3.6, kz + 0.12), M_POLE, [0.10, 0.25, 0.18])
        pr.prop(P_CLOCK, kx, 3.6, kz, 0.9, 0.45, ga=i, gb=j)
        pr.add_box((kx - 0.45, 3.6, kz - 0.45), (kx + 0.45, 4.5, kz + 0.45), M_KIOSK, [0.92, 0.90, 0.85, 2.0])
        pr.cur = -1
    else:              # a big tree on the corner
        _tree(pr, kx, kz, rng, big=True, kind=P_BIG_TREE, group=(i, j, 9))
    return kind, corner


def _crossing(pr, i, j, seed):
    """The landmark of crossing (i, j), traffic lights at crossings with an avenue, and a street-name sign."""
    lm = _landmark(pr, i, j, seed)
    c0, c1 = lcls(0, j, seed), lcls(1, i, seed)
    cx, cz = lpos(1, i, seed), lpos(0, j, seed)
    a = arms(i, j, seed)
    if MAIN in (c0, c1) and LANE not in (c0, c1):
        dx, dz = ROAD_OF[c1] + 0.9, ROAD_OF[c0] + 0.9
        for sx, sz, need in ((1, 1, ("E", "N")), (-1, 1, ("W", "N")), (-1, -1, ("W", "S")), (1, -1, ("E", "S"))):
            if any(a[n] for n in need):
                x, z = cx + sx * dx, cz + sz * dz
                pr.prop(P_TLIGHT, x, 0.0, z, 5.6, 0.0, 0.0, 1, i, j, 0)
                pr.add_box((x - 0.09, 0.0, z - 0.09), (x + 0.09, 5.6, z + 0.09), M_POLE, [0.25, 0.27, 0.28])
                pr.cur = -1
    if not (c0 == LANE and c1 == LANE):
        corner = ((lm[1] if lm else 0) + 2) % 4         # across from the landmark
        sx, sz = (1 if corner in (0, 3) else -1), (1 if corner in (0, 1) else -1)
        x = cx + sx * (ROAD_OF[c1] + (0.9 if c1 != LANE else 0.6))
        z = cz + sz * (ROAD_OF[c0] + (0.9 if c0 != LANE else 0.6))
        if MAIN in (c0, c1) and LANE not in (c0, c1):
            x, z = x + sx * 0.8, z + sz * 0.8           # clear of the traffic light on that corner
        pr.prop(P_SIGN, x, 0.0, z, 3.4, 0.0, 0.0, 1, i, j, 0)
        pr.add_box((x - 0.06, 0.0, z - 0.06), (x + 0.06, 3.4, z + 0.06), M_POLE, [0.25, 0.27, 0.28])
        pr.cur = -1


@lru_cache(maxsize=4096)
def block(i, j, seed=0):
    """Primitives of block (i, j), the street furniture of the two segments it owns (along x on line z = Z(j) and
    along z on line x = X(i), both between this block's corner lines), and crossing (i, j). Pure function."""
    pr = Prims()
    X0, X1, Z0, Z1, e, b = block_bounds(i, j, seed)
    kind = block_kind(i, j, seed)
    if kind == B_PARK:
        _park(pr, i, j, seed, e, b)
    elif kind == B_SQUARE:
        _square(pr, i, j, seed, e, b)
    elif kind == B_COMPOUND:
        _compound(pr, i, j, seed, e, b)
    elif kind == B_SCHOOL:
        _school(pr, i, j, seed, e, b)
    elif kind == B_MARKET:
        _market(pr, i, j, seed, e, b)
    else:
        _perimeter(pr, i, j, seed, e, b)
    _furniture(pr, 0, j, i, seed)
    _furniture(pr, 1, i, j, seed)
    _crossing(pr, i, j, seed)
    return pr.arrays()


def window(xmin, xmax, zmin, zmax, seed=0):
    """All primitives of the blocks overlapping a rectangle (and their crossings)."""
    i0, i1 = cell_of(1, xmin, seed) - 1, cell_of(1, xmax, seed) + 1
    j0, j1 = cell_of(0, zmin, seed) - 1, cell_of(0, zmax, seed) + 1
    parts = [block(i, j, seed) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
    out = [np.concatenate([p[k] for p in parts], 0) for k in range(4)]
    # prop indices are per block: shift them into the window's prop list
    off = np.cumsum([0] + [p[6].shape[0] for p in parts])
    for k in (4, 5):
        out.append(np.concatenate([np.where(p[k] >= 0, p[k] + o, -1) for p, o in zip(parts, off)]).astype(np.int32))
    out.append(np.concatenate([p[6] for p in parts], 0))
    return tuple(out)


def landmark_kind(i, j, seed=0):
    lm = _landmark(Prims(), i, j, seed)
    return None if lm is None else lm[0]
