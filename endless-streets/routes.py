"""Training routes: a home to start from and everyday destinations, reached through crossings with landmarks.

Pure functions of the seed, like the town. The home and destination buildings are ordinary town buildings whose
facade photo is generated with the home's name plate / the destination's sign (cityassets.facade asks special()).
Every turn of a route happens at a crossing that has a landmark, so each instruction can name one, and every step
of a route runs along a street segment that exists (the network has T-junctions, world.seg).

Crossings are (i, j) index pairs of the street network (world.py); directions are (dx, dz) unit steps between
neighbouring crossings. With the walker convention of server.py (forward (sin yaw, cos yaw), image right
(-cos yaw, sin yaw)) the right of a direction d is (-d.z, d.x).
"""
from functools import lru_cache

import world as Wd
import cityassets as CA

HOMES = ["幸福里小区", "和平里小区", "长青园小区", "安康家园", "丰收里小区", "惠民新村"]
DESTS = [("社区卫生服务中心", "白色门脸，门口有蓝色标识牌和红十字，玻璃门上贴着门诊时间"),
         ("农贸市场", "宽大的入口敞开着，里面能看到一排排菜摊和水果摊"),
         ("老年活动中心", "门口有两级台阶和不锈钢扶手，门边挂着活动安排的公告栏"),
         ("邮政所", "绿色门脸，门边立着一个绿色邮筒")]
NAMES_X = ["和平", "建设", "人民", "胜利", "光明", "幸福", "文化", "朝阳", "兴业", "春晖", "百花"]   # streets along x
NAMES_Z = ["解放", "长青", "东风", "团结", "新华", "育才", "康宁", "安居", "青年", "振兴", "友谊"]   # streets along z
SUFFIX = {0: {Wd.MAIN: "大道", Wd.STREET: "路", Wd.LANE: "巷"}, 1: {Wd.MAIN: "大街", Wd.STREET: "街", Wd.LANE: "胡同"}}
TURN_TO = {"left": "左转", "right": "右转", "straight": "直行"}


def street_name(fam, k, seed):
    """fam 0: the street along x on line z = Z(k); fam 1: the street along z on line x = X(k). The two directions
    use different words and suffixes by class (avenue / street / lane), and step 4 is coprime with 11, so 11
    neighbouring parallel streets never share a name."""
    names, salt = (NAMES_X, 61) if fam == 0 else (NAMES_Z, 62)
    n = len(names)
    return names[(k * 4 + Wd.h32(seed, salt) % n) % n] + SUFFIX[fam][Wd.lcls(fam, k, seed)]


def side_line(fam, k, seed):
    """Walking line: the middle of the sidewalk, metres from the street centre line."""
    return (Wd.lroad(fam, k, seed) + Wd.lhw(fam, k, seed)) / 2


def cpos(c, seed):
    return Wd.lpos(1, c[0], seed), Wd.lpos(0, c[1], seed)


def _street_of(a, b):
    """Street a step from crossing a to crossing b runs on: (fam, k)."""
    return (0, a[1]) if a[1] == b[1] else (1, a[0])


def _can_step(c, d, seed):
    i, j = c
    if d[1] == 0:
        return Wd.seg(0, j, i if d[0] > 0 else i - 1, seed)
    return Wd.seg(1, i, j if d[1] > 0 else j - 1, seed)


def _right(d):
    return (-d[1], d[0])


def _turn(d_in, d_out):
    s = d_out[0] * _right(d_in)[0] + d_out[1] * _right(d_in)[1]
    return "right" if s > 0 else ("left" if s < 0 else "straight")


def _fronts(seed, street, lo, hi):
    """Street-front buildings on a street between along-coordinates lo..hi: dicts rx (facade id), side (the side
    of the street the building stands on), along."""
    fam, k = street
    c = Wd.lpos(fam, k, seed)
    if fam == 0:
        box, boxp = Wd.window(lo, hi, c - 20, c + 20, seed)[:2]
    else:
        box, boxp = Wd.window(c - 20, c + 20, lo, hi, seed)[:2]
    out = []
    for b, p in zip(box, boxp):
        if b[6] != Wd.M_FACADE or int(round(float(p[6]))) // Wd.FKIND != Wd.F_SHOP:
            continue
        x0, _, z0, x1, _, z1 = (float(v) for v in b[:6])
        f = Wd.front_of(x0, z0, x1, z1, seed)
        if f is None or (f[2], f[3]) != (fam, k):
            continue
        along = (x0 + x1) / 2 if fam == 0 else (z0 + z1) / 2
        if lo <= along <= hi:
            out.append(dict(rx=tuple(int(round(v * 100)) for v in (x0, z0, x1, z1)), side=-f[1], along=along))
    return out


@lru_cache(maxsize=8)
def home(seed=0):
    """The home building on the street x = X(0), south of crossing (0, 0); the walk starts on the sidewalk in front
    of it, facing the crossing."""
    lo = Wd.lpos(0, -1, seed) + Wd.lhw(0, -1, seed) + 2
    hi = -Wd.lhw(0, 0, seed) - 6
    fr = [f for f in _fronts(seed, (1, 0), lo, hi) if f["side"] == 1] or _fronts(seed, (1, 0), lo, hi)
    f = min(fr, key=lambda f: abs(f["along"] + 18))
    name = HOMES[Wd.h32(seed, 63) % len(HOMES)]
    return dict(name=name, rx=f["rx"], x=f["side"] * side_line(1, 0, seed), z=f["along"], yaw=0.0,
                street=street_name(1, 0, seed))


def _plan(seed, k, taken, strict=2):
    """Route k: straight L1 blocks past crossing (0, 0), turn, L2 blocks, turn, L3 blocks, every step on an existing
    segment; both turns at crossings with landmarks; the destination is the last building on the right before the
    final crossing."""
    pref, base = ("right", "left")[k % 2], (1, 1, 2, 2)[k]
    shapes = [(L1, t1, L2, t2, L3) for L1 in range(1, 7) for t1 in ("right", "left") for L2 in (1, 2, 3)
              for t2 in ("right", "left") for L3 in (1, 2)]
    # preferred first turn (side, crossing) first, so the routes fan out; then a seed-dependent order
    shapes.sort(key=lambda sh: (sh[1] != pref, abs(sh[0] - base), Wd.h32(seed, k, sh[0], sh[2], sh[4], sh[3] == "right")))
    for L1, t1, L2, t2, L3 in shapes:
        d, cur, nodes = (0, 1), (0, 0), [(0, 0)]
        turns, ok = [], True
        for L, t in ((L1, t1), (L2, t2), (L3, None)):
            for _ in range(L):
                if not _can_step(cur, d, seed):
                    ok = False
                    break
                cur = (cur[0] + d[0], cur[1] + d[1])
                nodes.append(cur)
            if not ok:
                break
            if t is not None:
                turns.append(cur)
                r = _right(d)
                d = r if t == "right" else (-r[0], -r[1])
        if not ok:
            continue
        goal_x = nodes[-1]
        if goal_x in taken or (strict >= 2 and (turns[0], t1) in taken) or len(set(nodes)) != len(nodes):
            continue
        labels = [CA.landmark_label(i, j, seed) for i, j in turns]
        if None in labels or (strict >= 1 and len(set(labels)) < len(labels)):   # each turn its own landmark
            continue
        # destination: last building on the right of the final stretch, before the goal crossing
        a, b = nodes[-2], nodes[-1]
        street = _street_of(a, b)
        fam = street[0]
        pa, pb = (cpos(a, seed)[0], cpos(b, seed)[0]) if fam == 0 else (cpos(a, seed)[1], cpos(b, seed)[1])
        ia, ib = (a[0], b[0]) if fam == 0 else (a[1], b[1])
        hwa, hwb = Wd.lhw(1 - fam, ia, seed), Wd.lhw(1 - fam, ib, seed)
        lo, hi = (pa + hwa + 1, pb - hwb - 1) if pb > pa else (pb + hwb + 1, pa - hwa - 1)
        side = _right(d)[1] if fam == 0 else _right(d)[0]
        fr = [f for f in _fronts(seed, street, lo, hi) if f["side"] == side and f["rx"] not in taken]
        if not fr:
            continue
        f = max(fr, key=lambda f: f["along"] * (1 if pb > pa else -1))
        return nodes, d, street, side, f, (turns[0], t1)
    if strict:                  # a town with few landmarks: allow a repeated landmark name, then a shared first turn
        return _plan(seed, k, taken, strict - 1)
    raise RuntimeError(f"no route {k} for seed {seed}")


@lru_cache(maxsize=8)
def routes(seed=0):
    """Every route: name, destination, start pose, polyline, and one step per crossing passed (action, landmark,
    street entered)."""
    h = home(seed)
    taken, out = {h["rx"]}, []
    area = NAMES_Z[(Wd.h32(seed, 62) % len(NAMES_Z)) % len(NAMES_Z)]
    for k, (kind, desc) in enumerate(DESTS):
        nodes, d, street, side, f, first = _plan(seed, k, taken)
        taken |= {f["rx"], nodes[-1], first}           # routes differ in their first turn and their goal
        c = Wd.lpos(*street, seed)
        sl = side_line(*street, seed)
        gx, gz = (f["along"], c + side * sl) if street[0] == 0 else (c + side * sl, f["along"])
        cx, cz = (f["along"], c) if street[0] == 0 else (c, f["along"])
        pts = [[h["x"], h["z"]], [Wd.lpos(1, 0, seed), h["z"]]] + [list(cpos(n, seed)) for n in nodes] + [[cx, cz], [gx, gz]]
        pts.pop(-3)                                   # the goal crossing itself is not walked through
        steps = []
        din = (0, 1)
        for n, cr in enumerate(nodes[:-1]):
            dout = (nodes[n + 1][0] - cr[0], nodes[n + 1][1] - cr[1])
            act = _turn(din, dout)
            x, z = cpos(cr, seed)
            steps.append(dict(x=x, z=z, action=act, landmark=CA.landmark_label(*cr, seed),
                              street=street_name(*_street_of(cr, nodes[n + 1]), seed)))
            din = dout
        name = (area + "街" if kind.startswith("社区") else area) + kind
        steps.append(dict(x=gx, z=gz, action="arrive", side="right", landmark=name, street=None))
        out.append(dict(id=k, dest=name, kind=kind, desc=desc, rx=f["rx"], goal=[gx, gz],
                        start=[h["x"], h["z"]], yaw=h["yaw"], path=pts, steps=steps))
    return out


@lru_cache(maxsize=8)
def special(seed=0):
    """Facade id (rounded box corners, cm) -> ("home" | "dest", name, description) for cityassets.facade."""
    h = home(seed)
    out = {h["rx"]: ("home", h["name"], "")}
    for r in routes(seed):
        out[r["rx"]] = ("dest", r["dest"], r["desc"])
    return out


def blocks(x, z, r=300.0, seed=0):
    """Block outlines for the map: one rectangle per cell inside the street corridors (two cells joined by a
    missing street touch), [x0, z0, x1, z1], within r of (x, z)."""
    out = []
    for i in range(Wd.cell_of(1, x - r, seed), Wd.cell_of(1, x + r, seed) + 1):
        for j in range(Wd.cell_of(0, z - r, seed), Wd.cell_of(0, z + r, seed) + 1):
            b = Wd.block_bounds(i, j, seed)[5]
            out.append([round(v, 2) for v in (b[0], b[2], b[1], b[3])])
    return out


def public(seed=0):
    """What the page needs: home, routes (the map's blocks come from /blocks)."""
    h = home(seed)
    return dict(home=dict(name=h["name"], x=h["x"], z=h["z"]),
                routes=[{k: v for k, v in r.items() if k not in ("rx", "desc")} for r in routes(seed)])


if __name__ == "__main__":
    import json
    print(json.dumps(public(0), ensure_ascii=False, indent=1))
