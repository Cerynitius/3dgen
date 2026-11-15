"""Headless walk: a fixed route through the town, timing per view and frames saved for inspection.
usage: walk_test.py <out dir> [server args...]"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0]] + sys.argv[2:]
import server as S  # noqa: E402

torch.cuda.set_device(S.WD)
pipe = S.Pipeline(S.store)
S.pipe = pipe
w = S.Walker()
# route: walk north along x=0 through the crossing at (0,0), then turn east at (0,40) toward (30,40)
route = [(0.0, -12.0 + 0.7 * k, 0.0) for k in range(75)] + [(0.7 * k, 40.5, math.pi / 2) for k in range(1, 45)]
# warm-up: build everything the route needs
need = set()
for x, z, _ in route[::5]:
    need |= set(S.ring(x, z, S.args.radius))
S.store.request(sorted(need, key=lambda t: abs(t[0]) + abs(t[1])))
t0 = time.time()
while len(S.store.ready) < len(need):
    time.sleep(0.5)
    if time.time() - t0 > 900:
        break
print(f"[walk_test] {len(S.store.ready)}/{len(need)} tiles in {time.time() - t0:.0f} s, build ms median {np.median(S.store.build_ms):.0f}", flush=True)
res = {}
for view in (1, 2, 3):
    if pipe.fixer is not None:
        pipe.fixer.reset()
    ms, frames = [], []
    for k, (x, z, yaw) in enumerate(route):
        w.x, w.z, w.yaw = x, z, yaw + 0.35 * math.sin(k / 9.0)
        torch.cuda.synchronize()
        t = time.perf_counter()
        o, _, _ = pipe.frame(w, view)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t) * 1000)
        if k % 15 == 7:
            frames.append(o["img"].cpu().numpy())
    res[view] = frames
    print(f"[walk_test] view {view}: median {np.median(ms[5:]):.1f} ms, p90 {np.percentile(ms[5:], 90):.1f} ms, splats now {pipe.cat["means"].shape[0]/1e6:.2f} M", flush=True)
rows = [np.concatenate([res[1][i], res[2][i], res[3][i]], 1) for i in range(len(res[1]))]
for i, r in enumerate(rows):
    Image.fromarray((r * 255).astype(np.uint8)).save(out / f"triple_{i}.jpg", quality=90)
print(f"[walk_test] {len(rows)} triples -> {out}", flush=True)
