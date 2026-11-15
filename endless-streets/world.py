"""Endless procedural town: a street grid with buildings, shop signs, street furniture and
landmarks at some crossings.

Everything is a pure function of (seed, block index): a block is generated from its own
hash and never looks at its neighbours, so any window of the town can be built on its own
and a place you walk back to is the place you left.

Frame: metres, Y up, ground at y = 0. Street centrelines run along x = i*P and z = j*P;
crossings sit at (i*P, j*P). Block (i, j) fills [i*P + HW, (i+1)*P - HW] in x and z.

Primitives are axis-aligned boxes and spheres, each with a material id and eight float
parameters the shader reads (colours, window spacing, sign colour, a per-object seed).
"""
import math
from functools import lru_cache

import numpy as np

P = 40.0            # block period (crossing to crossing)
ROAD = 3.5          # half width of the carriageway
HW = 6.0            # half width of the street corridor (carriageway + sidewalk)
GF = 4.2            # ground-floor height (shop front + sign band)
FH = 3.2            # upper floor height
EYE = 1.6           # eye height of a walker

# materials
M_FACADE, M_ROOF, M_TRUNK, M_CROWN, M_POLE, M_LAMP, M_KIOSK, M_STONE, M_ADCOL, M_BENCH, M_BRONZE = range(1, 12)
M_GROUND = 0        # the y = 0 plane (asphalt / sidewalk / markings), not a primitive
SKY = -1

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


# Props: things whose look is generated while walking (cityassets). Columns of the props array:
# kind, x, y0, z, height, half width, angle (rad, of the billboard plane), crossed (0/1), group a, b, c
P_STREET_TREE, P_PARK_TREE, P_BIG_TREE, P_STATUE, P_BENCH, P_KIOSK, P_ADCOL, P_CLOCK = range(1, 9)
BILLBOARD_PROPS = (P_STREET_TREE, P_PARK_TREE, P_BIG_TREE, P_STATUE, P_BENCH)   # drawn as generated cut-outs
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


def _building(pr, rng, x0, z0, x1, z1, face_axis, seed):
    """One building box; face_axis tells which pair of faces fronts a street (shader uses world coords anyway)."""
    floors = 1 + rng.i(6) + (rng.i(3) == 0) * rng.i(3)
    h = GF + floors * FH + rng.r(0.0, 0.6)
    wall = PALETTE_WALL[rng.i(len(PALETTE_WALL))] * rng.r(0.9, 1.06)
    sign = PALETTE_SIGN[rng.i(len(PALETTE_SIGN))]
    ws = rng.r(2.3, 3.6)          # window spacing
    style = rng.i(3)              # 0 plaster, 1 brick, 2 panel
    bay = rng.r(2.8, 4.5)         # shop bay width
    pr.add_box((x0, 0.0, z0), (x1, h, z1), M_FACADE,
               [*wall, ws, style + 4 * rng.i(len(PALETTE_SIGN)), bay, float(seed % 100000), h])
    return h


def _tree(pr, x, z, rng, big=False, kind=P_STREET_TREE, group=(0, 0, 0)):
    s = 1.6 if big else 1.0
    pr.prop(kind, x, 0.0, z, 0.0, 0.0, 0.0, 1, *group)     # size and angle are set by cityassets
    pr.add_box((x - 0.14 * s, 0.0, z - 0.14 * s), (x + 0.14 * s, 2.6 * s, z + 0.14 * s), M_TRUNK, [0.36, 0.27, 0.20])
    g = np.array([0.22, 0.42, 0.18], np.float32) * rng.r(0.8, 1.15)
    pr.add_sphere((x, 3.6 * s, z), 1.5 * s, M_CROWN, [*g, rng.u() * 1000])
    pr.add_sphere((x + 0.6 * s, 3.0 * s, z - 0.4 * s), 1.0 * s, M_CROWN, [*(g * 0.9), rng.u() * 1000])
    pr.cur = -1


def _lamp(pr, x, z, ax, sgn):
    pr.add_box((x - 0.07, 0.0, z - 0.07), (x + 0.07, 5.2, z + 0.07), M_POLE, [0.20, 0.22, 0.22])
    dx, dz = (sgn * 0.7, 0.0) if ax == 0 else (0.0, sgn * 0.7)
    cx, cz = x + dx, z + dz
    pr.add_box((min(x, cx) - 0.06, 5.05, min(z, cz) - 0.06), (max(x, cx) + 0.06, 5.2, max(z, cz) + 0.06), M_POLE, [0.20, 0.22, 0.22])
    pr.add_box((cx - 0.22, 4.85, cz - 0.22), (cx + 0.22, 5.05, cz + 0.22), M_LAMP, [0.90, 0.88, 0.75])


def _landmark(pr, i, j, seed):
    """Landmarks at some crossings: what a route is remembered by."""
    rng = Rng(seed, 77, i, j)
    if rng.u() > 0.55:
        return None
    cx, cz = i * P, j * P
    kind = rng.i(5)
    col = PALETTE_BRIGHT[rng.i(len(PALETTE_BRIGHT))]
    corner = rng.i(4)
    sx, sz = (1 if corner in (0, 3) else -1), (1 if corner in (0, 1) else -1)
    kx, kz = cx + sx * (HW - 1.3), cz + sz * (HW - 1.3)     # on the sidewalk corner
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
    return kind


@lru_cache(maxsize=4096)
def block(i, j, seed=0):
    """Primitives of block (i, j) and the sidewalk furniture along its edges, plus the landmark of
    crossing (i, j). Pure function of its arguments."""
    pr = Prims()
    rng = Rng(seed, i, j)
    bx0, bz0 = i * P + HW, j * P + HW
    bx1, bz1 = (i + 1) * P - HW, (j + 1) * P - HW
    L = bx1 - bx0
    park = rng.u() < 0.08
    if park:
        for k in range(5):
            _tree(pr, rng.r(bx0 + 2, bx1 - 2), rng.r(bz0 + 2, bz1 - 2), rng, big=rng.u() < 0.5,
                  kind=P_PARK_TREE, group=(i, j, 8))
        pr.prop(P_BENCH, bx0 + 10.5, 0.0, bz0 + 12.25, 0.0, 1.5, 0.0, 0, i, j, 0)
        pr.add_box((bx0 + 9, 0.0, bz0 + 12), (bx0 + 12, 0.45, bz0 + 12.5), M_BENCH, [0.45, 0.30, 0.18])
        pr.cur = -1
    else:
        # perimeter block: lots along each side; N/S sides run full length, E/W sides fill between them
        dn, ds = rng.r(9, 12), rng.r(9, 12)
        sides = [("S", bz0), ("N", bz1)]
        for name, zf in sides:
            u = bx0
            k = 0
            while u < bx1 - 0.5:
                w = min(rng.r(5.5, 11.0), bx1 - u)
                if bx1 - u - w < 4.0:
                    w = bx1 - u
                d = dn if name == "N" else ds
                z0, z1 = (zf - d, zf) if name == "N" else (zf, zf + d)
                _building(pr, rng, u, z0, u + w, z1, 2, h32(seed, i, j, k, 1))
                u += w; k += 1
        for name, xf in (("W", bx0), ("E", bx1)):
            u = bz0 + ds
            k = 0
            while u < bz1 - dn - 0.5:
                w = min(rng.r(5.5, 11.0), bz1 - dn - u)
                if bz1 - dn - u - w < 4.0:
                    w = bz1 - dn - u
                d = rng.r(8, 11)
                x0, x1 = (xf, xf + d) if name == "W" else (xf - d, xf)
                _building(pr, rng, x0, u, x1, u + w, 0, h32(seed, i, j, k, 2))
                u += w; k += 1
    # street furniture on the sidewalks of the two streets this block owns: the street along z = j*P
    # (south of the block row j) and along x = i*P (west of block column i), between crossings
    for ax in (0, 1):
        for side in (-1, 1):
            for k in range(2):
                t = (k + 0.5) * P / 2 + (rng.r(-2, 2))
                off = side * (ROAD + 1.6)
                if ax == 0:   # street along x at z = j*P
                    x, z = i * P + t, j * P + off
                else:         # street along z at x = i*P
                    x, z = i * P + off, j * P + t
                if HW + 1 < t < P - HW - 1:
                    if (k + (side > 0)) % 2 == 0:
                        _lamp(pr, x, z, 1 - ax, -side)
                    elif rng.u() < 0.7:
                        _tree(pr, x, z, rng, group=(i, j, ax))   # one species per street segment
    _landmark(pr, i, j, seed)
    # sidewalk benches (own random stream, so the town above is unchanged): facing the street, by the shop fronts
    rb = Rng(seed, i, j, 55)
    for ax in (0, 1):
        if rb.u() < 0.5:
            t, side = rb.r(HW + 4, P - HW - 4), (1 if rb.u() < 0.5 else -1)
            off = side * (HW - 0.45)
            if ax == 0:
                x, z = i * P + t, j * P + off
                lo, hi = (x - 0.9, 0.0, z - 0.25), (x + 0.9, 0.45, z + 0.25)
                ang = 0.0
            else:
                x, z = i * P + off, j * P + t
                lo, hi = (x - 0.25, 0.0, z - 0.9), (x + 0.25, 0.45, z + 0.9)
                ang = math.pi / 2
            pr.prop(P_BENCH, x, 0.0, z, 0.0, 0.9, ang, 0, i, j, 1 + ax)
            pr.add_box(lo, hi, M_BENCH, [0.45, 0.30, 0.18])
            pr.cur = -1
    return pr.arrays()


def window(xmin, xmax, zmin, zmax, seed=0):
    """All primitives of the blocks overlapping a rectangle (and their crossings)."""
    i0, i1 = math.floor(xmin / P) - 1, math.floor(xmax / P) + 1
    j0, j1 = math.floor(zmin / P) - 1, math.floor(zmax / P) + 1
    parts = [block(i, j, seed) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
    out = [np.concatenate([p[k] for p in parts], 0) for k in range(4)]
    # prop indices are per block: shift them into the window's prop list
    off = np.cumsum([0] + [p[6].shape[0] for p in parts])
    for k in (4, 5):
        out.append(np.concatenate([np.where(p[k] >= 0, p[k] + o, -1) for p, o in zip(parts, off)]).astype(np.int32))
    out.append(np.concatenate([p[6] for p in parts], 0))
    return tuple(out)


def landmark_kind(i, j, seed=0):
    return _landmark(Prims(), i, j, seed)
