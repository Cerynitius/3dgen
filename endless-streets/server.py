"""Walk server: an endless generated town, reconstructed tile by tile and repaired live by FastFix3D.

  build GPU  : tiles are baked on demand (ray-traced rig -> one surface disc per pixel -> pruned), nearest first
  walk GPU   : the loaded tiles are splatted at the walker's pose, sky + haze composited, then FastFix3D repairs
               the frame against the nearest rig view (the tile's "photo") and the frame goes out as JPEG

The browser sends its input state over a WebSocket; the server moves the walker (with collision against the town)
and streams frames back. Views: 1 3DGS render, 2 FastFix3D, 3 ground truth (ray tracer), 4 render | FastFix3D.

usage: server.py [--port 8095] [--seed 0] [--walk-gpu 0] [--build-gpu 1] [--radius 2] [--no-fix]
"""
import argparse
import asyncio
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.path.expanduser("~/src/ultrafix/src"))

import raytrace as RT  # noqa: E402
import splatrender as SR  # noqa: E402
import stiles as ST  # noqa: E402
import world as Wd  # noqa: E402
import cityassets as CA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8095)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--walk-gpu", type=int, default=0)
ap.add_argument("--build-gpus", default="1,2")
ap.add_argument("--radius", type=int, default=3)
ap.add_argument("--lod", default="0.025,0.06,0.15", help="voxel size of the three detail levels (m)")
ap.add_argument("--lod-dist", default="8,25", help="distance (m) at which a tile drops to level 1, 2")
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--height", type=int, default=540)
ap.add_argument("--fov", type=float, default=90.0)
ap.add_argument("--engine", default=str(HERE / "runs/fast_960x540.engine"))
ap.add_argument("--weights", default=os.path.expanduser("~/src/ultrafix/weights/ultrafix_fast.safetensors"))
ap.add_argument("--no-fix", action="store_true")
ap.add_argument("--recon", choices=("bake", "worldmirror"), default="bake",
                help="bake: splats straight from the ray-traced rig pixels; worldmirror: the feed-forward model's splats")
ap.add_argument("--cache", default=str(HERE / "cache"))
ap.add_argument("--quality", type=int, default=80)
ap.add_argument("--assets", default="http://127.0.0.1:8097", help="facade generator (assetgen.py)")
ap.add_argument("--no-assets", action="store_true", help="procedural facades only (no generated photos)")
args = ap.parse_args()

WD = f"cuda:{args.walk_gpu}"
BDS = [f"cuda:{g}" for g in args.build_gpus.split(",") if g != ""]
LOD = [float(v) for v in args.lod.split(",")]
LOD_D = [float(v) for v in args.lod_dist.split(",")]
W, H = args.width, args.height
K = RT.K_of(W, H, math.radians(args.fov))
if args.recon == "bake":
    SPEC = ST.Spec(bake=True, bake_sigma=0.3, bake_voxel=LOD[0])
    CACHE = Path(args.cache) / f"v3_seed{args.seed}_{SPEC.bake_voxel}_{SPEC.bake_sigma}"
else:
    # the labyrinth's reconstruction path: model positions (gs_depth x one scale per view fitted on the true
    # depth), model colours / scales / opacities, sky pixels and off-surface splats dropped, voxel-pruned
    SPEC = ST.Spec(bake=False, positions="model", appearance="model", sky_mask=True, max_splats=1_200_000, ss=1, down_pitch=0.0)
    CACHE = Path(args.cache) / f"wm3_seed{args.seed}_{SPEC.max_splats}"
if args.no_assets:
    RT.TEXTURES = False
else:
    CACHE = CACHE.with_name(CACHE.name + f"_{CA.VERSION}_all")   # facades + props + ground generated
CACHE.mkdir(parents=True, exist_ok=True)
POS = [0.0, -12.0]          # last walker position (asset priorities)


class Assets:
    """Asks the facade generator for what is needed, in order: the buildings standing in the tiles about to be
    built, then the buildings around the walker (ground truth and reference views). Tiles wait for their own."""
    def __init__(self, store):
        import urllib.request
        self.store = store
        self.op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.tiles = {}
        self.err = None
        threading.Thread(target=self._loop, daemon=True).start()

    def tile_facades(self, t):
        if t not in self.tiles:
            o, h = ST.origin(*t), ST.TILE / 2
            # a tile also owns half of each neighbouring block interior (stiles.owned)
            self.tiles[t] = CA.assets_in(o[0] - h, o[0] + h, o[2] - h, o[2] + h, args.seed, margin=h + 0.5)
        return self.tiles[t]

    def demand(self):
        with self.store.lock:
            tiles = list(self.store.pending) + list(self.store.want)
        jobs, seen = [], set()

        def add(f):
            if f["key"] not in seen and not CA.have(args.seed, f["key"]):
                seen.add(f["key"])
                jobs.append({k: f[k] for k in ("key", "prompt", "w", "h", "gseed", "kind")})
        for t in tiles[:10]:
            for f in self.tile_facades(t):
                add(f)
        x, z = POS
        near = CA.assets_in(x - 45, x + 45, z - 45, z + 45, args.seed)
        near.sort(key=lambda f: (f["cx"] - x) ** 2 + (f["cz"] - z) ** 2)
        for f in near:
            add(f)
        return jobs

    def _loop(self):
        import urllib.request
        while True:
            try:
                body = json.dumps({"client": f"walk{args.port}", "seed": args.seed, "jobs": self.demand()}).encode()
                self.op.open(urllib.request.Request(args.assets + "/want", data=body,
                                                    headers={"Content-Type": "application/json"}), timeout=5).read()
                self.err = None
            except Exception as e:
                if str(e) != self.err:
                    print(f"[assets] {type(e).__name__}: {e}", flush=True)
                self.err = str(e)
            time.sleep(1.0)

    def wait(self, t, timeout=300.0):
        keys = [f["key"] for f in self.tile_facades(t)]
        t0 = time.time()
        while time.time() - t0 < timeout:
            miss = [k for k in keys if not CA.have(args.seed, k)]
            if not miss:
                return 0
            time.sleep(0.3)
        print(f"[assets] tile {t}: {len(miss)} facades still missing after {timeout:.0f} s, built procedural", flush=True)
        return len(miss)


# ------------------------------------------------------------------ tiles
class TileStore:
    """Tiles built on the build GPU in a worker thread, nearest to the walker first; finished tiles move to the
    walk GPU. Each tile is a pure function of (seed, a, b), so the disk cache never goes stale."""

    def __init__(self):
        self.ready = {}            # (a, b) -> splat dict on the walk GPU
        self.rig = {}              # (a, b) -> world c2w [N,4,4] (numpy)
        self.want = []
        self.pending = set()
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.build_ms = []
        self.busy = 0
        self.infer = None
        for dev in (BDS[:1] if args.recon == "worldmirror" else BDS):   # one resident model
            threading.Thread(target=self._worker, args=(dev,), daemon=True).start()

    def request(self, tiles):
        with self.cv:
            self.want = [t for t in tiles if t not in self.ready]
            self.cv.notify()

    def _worker(self, dev):
        torch.cuda.set_device(dev)
        while True:
            with self.cv:
                while not self.want:
                    self.cv.wait()
                t = self.want.pop(0)
                if t in self.ready or t in self.pending:
                    continue
                self.pending.add(t)
                self.busy += 1
            try:
                self._load(t, dev)
            finally:
                with self.lock:
                    self.pending.discard(t)
                    self.busy -= 1

    def _load(self, t, BD):
        if True:
            a, b = t
            f = CACHE / f"tile_{a}_{b}.pt"
            t0 = time.perf_counter()
            if f.exists():
                g = torch.load(f, map_location="cpu")
            else:
                if ASSETS is not None:
                    ASSETS.wait(t)
                if args.recon == "worldmirror" and self.infer is None:
                    self.infer = ST.load_pipeline()      # resident on this worker's GPU
                g, rep = ST.build(a, b, self.infer, SPEC, seed=args.seed, device=BD)
                g = {k: v.half().cpu() if v.is_floating_point() and k not in ("means", "rig_c2w", "rig_K") else v.cpu() for k, v in g.items()}
                torch.save(g, f)
            o = torch.tensor(ST.origin(a, b), device=WD)
            gw = {k: g[k].to(WD).float() for k in ("means", "scales", "quats", "rgb", "opacity")}
            gw["means"] = gw["means"] + o
            # one splat survives per voxel, so its disc must at least span the voxel or the surface shows gaps
            if args.recon == "bake":
                gw["scales"][:, :2] = gw["scales"][:, :2].clamp(min=0.45 * LOD[0])
            levels = [gw] + [coarsen(gw, v, all_axes=args.recon != "bake") for v in LOD[1:]]
            rig = g["rig_c2w"].numpy().copy()
            rig[:, :3, 3] += ST.origin(a, b)
            with self.lock:
                self.ready[t] = levels
                self.rig[t] = rig
            self.build_ms.append((time.perf_counter() - t0) * 1000)

    def drop_far(self, keep):
        with self.lock:
            for t in list(self.ready):
                if t not in keep:
                    del self.ready[t], self.rig[t]


def coarsen(g, vs, all_axes=False):
    """A coarser level of a tile: one splat per voxel of size vs, at the sharpest member's place and
    orientation, carrying the voxel's mean colour and widened in-plane to cover the voxel."""
    q = torch.floor(g["means"] / vs).long()
    q -= q.min(0).values
    m = q.max(0).values + 1
    key = (q[:, 0] * m[1] + q[:, 1]) * m[2] + q[:, 2]
    uk, inv = torch.unique(key, return_inverse=True)
    n = uk.numel()
    cnt = torch.zeros(n, device=key.device).index_add_(0, inv, torch.ones_like(inv, dtype=torch.float32))
    col = torch.zeros(n, 3, device=key.device).index_add_(0, inv, g["rgb"]) / cnt[:, None]
    smin = g["scales"][:, :2].min(1).values
    order = torch.argsort(smin)
    first = torch.full((n,), -1, dtype=torch.long, device=key.device)
    first.scatter_(0, inv[order].flip(0), order.flip(0))     # last write wins -> the smallest-scale member
    out = {k: v[first] for k, v in g.items()}
    out["rgb"] = col
    sc = out["scales"].clone()
    if all_axes:     # model splats have no surface frame: widen every axis
        sc = sc.clamp(min=0.5 * vs)
    else:
        sc[:, :2] = sc[:, :2].clamp(min=0.5 * vs)
    out["scales"] = sc
    return out


def ring(x, z, r):
    ca, cb = round(x / ST.TILE), round(z / ST.TILE)
    out = [(ca + i, cb + j) for i in range(-r, r + 1) for j in range(-r, r + 1) if ST.kind(ca + i, cb + j) != "interior"]
    out.sort(key=lambda t: (t[0] * ST.TILE - x) ** 2 + (t[1] * ST.TILE - z) ** 2)
    return out


# ------------------------------------------------------------------ walker
class Walker:
    def __init__(self):
        self.x, self.z, self.yaw, self.pitch = 0.0, -12.0, 0.0, 0.0
        self.y = Wd.EYE
        self.solid = None
        self.solid_at = None

    def _obstacles(self):
        key = (round(self.x / 20), round(self.z / 20))
        if key != self.solid_at:
            box, _, sph, _ = Wd.window(self.x - 30, self.x + 30, self.z - 30, self.z + 30, args.seed)[:4]
            b = box[(box[:, 4] > 0.3) & (box[:, 6] != Wd.M_LAMP)]
            self.solid = b[:, [0, 2, 3, 5]]
            self.solid_at = key

    def blocked(self, x, z, r=0.3):
        self._obstacles()
        s = self.solid
        return bool(np.any((x > s[:, 0] - r) & (x < s[:, 2] + r) & (z > s[:, 1] - r) & (z < s[:, 3] + r)))

    def step(self, inp, dt):
        self.yaw += inp.get("dyaw", 0.0)
        self.pitch = float(np.clip(self.pitch + inp.get("dpitch", 0.0), -1.2, 1.2))
        f = (inp.get("w", 0) - inp.get("s", 0))
        r = (inp.get("d", 0) - inp.get("a", 0))
        sp = 3.2 if inp.get("shift") else 1.4
        fx, fz = math.sin(self.yaw), math.cos(self.yaw)
        rx, rz = -math.cos(self.yaw), math.sin(self.yaw)   # image right in this convention
        dx = (f * fx + r * rx) * sp * dt
        dz = (f * fz + r * rz) * sp * dt
        if not self.blocked(self.x + dx, self.z):
            self.x += dx
        if not self.blocked(self.x, self.z + dz):
            self.z += dz

    def c2w(self):
        return RT.c2w(self.yaw, self.pitch, (self.x, self.y, self.z))


# ------------------------------------------------------------------ frame pipeline
class Pipeline:
    def __init__(self, store):
        torch.cuda.set_device(WD)
        self.store = store
        self.Kt = torch.tensor(K, device=WD)
        xs = torch.arange(W, device=WD).float() + 0.5
        ys = torch.arange(H, device=WD).float() + 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.cam = torch.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], torch.ones_like(xx)], -1).reshape(-1, 3)
        self.cam = torch.nn.functional.normalize(self.cam, dim=-1)
        self.fixer = None
        if not args.no_fix and Path(args.engine).exists():
            from ultrafix.fast import load_fast
            from ultrafix.realtime import RealtimeFixer
            self.fixer = RealtimeFixer(args.engine, load_fast(args.weights, dtype=torch.float32))
            self.fixer.refs.max_items = 24
        self.refs = {}             # rig key -> (img, depth, viewmat, K) at walk resolution
        self.ref_scene = None
        self.gt_scene = None
        self.last_ref = None
        self.cat = None
        self.cat_keys = None

    def scene_near(self, x, z, cur):
        if cur is None or abs(cur.cx - x) > 40 or abs(cur.cz - z) > 40:
            cur = RT.Scene(x - 130, x + 130, z - 130, z + 130, seed=args.seed, device=WD)
            cur.cx, cur.cz = x, z
            cur.t_refresh = time.time()
        elif time.time() - cur.t_refresh > 2.0:
            cur.refresh_textures()
            cur.t_refresh = time.time()
        return cur

    def splats(self, walker):
        """Visible tiles at a level of detail by distance; the concatenation is cached until the choice changes."""
        fx, fz = math.sin(walker.yaw), math.cos(walker.yaw)
        half = math.radians(args.fov) / 2 * W / max(W, H) + math.radians(20)
        with self.store.lock:
            choice = []
            for t in self.store.ready:
                ox, oz = t[0] * ST.TILE, t[1] * ST.TILE
                h = ST.TILE      # owned area reaches half a tile into the block interiors (stiles.owned)
                dx = max(abs(walker.x - ox) - h, 0.0)
                dz = max(abs(walker.z - oz) - h, 0.0)
                d = math.hypot(dx, dz)
                if d > 0:
                    vis = False
                    for cx, cz in ((ox - h, oz - h), (ox + h, oz - h), (ox - h, oz + h), (ox + h, oz + h), (ox, oz)):
                        vx, vz = cx - walker.x, cz - walker.z
                        ang = math.acos(max(-1.0, min(1.0, (vx * fx + vz * fz) / max(math.hypot(vx, vz), 1e-6))))
                        if ang < half:
                            vis = True
                            break
                    if not vis and d > 4:
                        continue
                dl = math.hypot(max(abs(walker.x - ox) - ST.TILE / 2, 0.0), max(abs(walker.z - oz) - ST.TILE / 2, 0.0))
                lvl = 0 if dl < LOD_D[0] else (1 if dl < LOD_D[1] else 2)
                choice.append((t, lvl))
            choice = tuple(sorted(choice))
            if choice != self.cat_keys:
                self.cat = {k: torch.cat([self.store.ready[t][l][k] for t, l in choice]) for k in ("means", "scales", "quats", "rgb", "opacity")} if choice else None
                self.cat_keys = choice
            keys = tuple(sorted(self.store.ready))
            if keys != getattr(self, "rig_set", None):
                self.rig_set = keys
                self.train = torch.tensor(np.concatenate([self.store.rig[t] for t in keys]), device=WD) if keys else None
                self.rig_keys = [(t, i) for t in keys for i in range(self.store.rig[t].shape[0])]
                self.rig_index = {k: n for n, k in enumerate(self.rig_keys)}
        return self.cat

    def pick_ref(self, c2w):
        """The rig view nearest the walker that looks the same way (a captured frame of the walk)."""
        tc = self.train
        p = torch.tensor(c2w[:3, 3], device=WD)
        f = torch.tensor(c2w[:3, 2], device=WD)
        d = (tc[:, :3, 3] - p).norm(dim=-1)
        cosang = (tc[:, :3, 2] * f).sum(-1)
        score = d + 6.0 * (1 - cosang) + 100.0 * (cosang < 0.3)
        i = int(score.argmin())
        # hysteresis: a new reference costs a pass of the reference branch, so keep the current one
        # unless another view is clearly closer
        if self.last_ref is not None and self.last_ref in self.rig_index:
            j = self.rig_index[self.last_ref]
            if float(score[j]) < float(score[i]) + 1.5:
                i = j
        key = self.rig_keys[i]
        if key not in self.refs:
            self.ref_scene = self.scene_near(float(p[0]), float(p[2]), self.ref_scene)
            cw = self.train[i].cpu().numpy()
            img, z = RT.render(RT.Raytracer(self.ref_scene), cw, K, W, H, haze=False, ss=1)
            img = RT.add_haze(img, z, (z > 0).float())
            img = torch.where((z > 0)[..., None], img, self._sky(torch.tensor(cw, device=WD)))
            self.refs[key] = (img.contiguous(), z.contiguous(), torch.linalg.inv(self.train[i]).contiguous(), self.Kt)
            if len(self.refs) > 64:
                self.refs.pop(next(iter(self.refs)))
        return key, self.refs[key]

    def _sky(self, c2w_t):
        d = self.cam @ c2w_t[:3, :3].T
        return RT.sky(d).reshape(H, W, 3).pow(1 / 1.15)

    @torch.no_grad()
    def frame(self, walker, view):
        c2w = walker.c2w()
        c2w_t = torch.tensor(c2w, device=WD)
        vm = torch.linalg.inv(c2w_t)
        out = {}
        t0 = time.perf_counter()
        g = self.splats(walker)
        sky = self._sky(c2w_t)
        if view == 3 or g is None:
            self.gt_scene = self.scene_near(walker.x, walker.z, self.gt_scene)
            img, z = RT.render(RT.Raytracer(self.gt_scene), c2w, K, W, H, haze=False, ss=1)
            img = RT.add_haze(img, z, (z > 0).float())
            img = torch.where((z > 0)[..., None], img, sky)
            out["img"] = img
            return out, (time.perf_counter() - t0) * 1000, None
        img, depth, alpha = SR.render(g, vm, self.Kt, W, H)
        img = img + (1 - alpha[..., None]) * sky
        img = RT.add_haze(img, depth, alpha)
        raw = img.clamp(0, 1)
        if view in (2, 4) and self.fixer is not None:
            key, (rimg, rdep, rvm, rK) = self.pick_ref(c2w)
            feat = self.fixer.features(depth, alpha, vm, self.Kt, rimg, rdep, rvm, rK, self.train)
            fixed = self.fixer.fix(raw, depth, feat, vm, self.Kt, rimg, key)
            if view == 2:
                out["img"] = fixed.clamp(0, 1)
            else:
                half = W // 2
                out["img"] = torch.cat([raw[:, W // 4:W // 4 + half], fixed.clamp(0, 1)[:, W // 4:W // 4 + half]], 1)
            self.last_ref = key
        else:
            out["img"] = raw
        return out, (time.perf_counter() - t0) * 1000, self.last_ref


# ------------------------------------------------------------------ server
from aiohttp import web, WSMsgType  # noqa: E402
from torchvision.io import encode_jpeg  # noqa: E402

store = TileStore()
ASSETS = None if args.no_assets else Assets(store)
pipe = None


async def index(_):
    return web.FileResponse(HERE / "client/walk.html")


async def ws_handler(request):
    global pipe
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    walker = Walker()
    inp, view = {}, 2 if pipe.fixer is not None else 1
    if pipe.fixer is not None:
        pipe.fixer.reset()
    last = time.perf_counter()
    alive = True
    fps_t, fps_n, fps = time.perf_counter(), 0, 0.0

    async def reader():
        nonlocal inp, view, alive
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                m = json.loads(msg.data)
                if "view" in m:
                    view = int(m["view"])
                    if pipe.fixer is not None:
                        pipe.fixer.reset()
                for k in ("w", "a", "s", "d", "shift"):
                    if k in m:
                        inp[k] = m[k]
                inp["dyaw"] = inp.get("dyaw", 0.0) + m.get("dyaw", 0.0)
                inp["dpitch"] = inp.get("dpitch", 0.0) + m.get("dpitch", 0.0)
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
        alive = False

    task = asyncio.ensure_future(reader())
    try:
        while alive and not ws.closed:
            now = time.perf_counter()
            dt = min(now - last, 0.1)
            last = now
            walker.step(inp, dt)
            POS[0], POS[1] = walker.x, walker.z
            inp["dyaw"] = inp["dpitch"] = 0.0
            want = ring(walker.x, walker.z, args.radius)
            store.request(want)
            store.drop_far(set(ring(walker.x, walker.z, args.radius + 1)))
            out, ms, ref = pipe.frame(walker, view)
            jpg = encode_jpeg((out["img"].permute(2, 0, 1) * 255).to(torch.uint8), quality=args.quality).cpu().numpy().tobytes()
            fps_n += 1
            if now - fps_t > 1.0:
                fps, fps_t, fps_n = fps_n / (now - fps_t), now, 0
            hud = {"x": round(walker.x, 1), "z": round(walker.z, 1), "view": view, "ms": round(ms, 1), "fps": round(fps, 1),
                   "tiles": len(store.ready), "queue": len(store.want) + store.busy, "fix": pipe.fixer is not None,
                   "splats": round(pipe.cat["means"].shape[0] / 1e6, 2) if pipe.cat is not None else 0,
                   "build_ms": round(float(np.median(store.build_ms[-20:])), 0) if store.build_ms else None}
            await ws.send_bytes(json.dumps(hud).encode().ljust(256) + jpg)
            await asyncio.sleep(0)
    finally:
        task.cancel()
    return ws


def main():
    global pipe
    torch.cuda.set_device(WD)
    pipe = Pipeline(store)
    store.request(ring(0.0, -12.0, args.radius))
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    print(f"[walk] http://0.0.0.0:{args.port}/  walk {WD}, build {BDS}, fix={'on' if pipe.fixer else 'off'}", flush=True)
    web.run_app(app, port=args.port, print=None)


if __name__ == "__main__":
    main()
