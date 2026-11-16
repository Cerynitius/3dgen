"""Near-field tracer in a process of its own (the tracer is launch-bound Python, and in the walk process it held the GIL
the frames need): it keeps its own ray-tracing window around the walker, traces the latest requested pose straight
from the generated images and leaves image, z-depth and pose in one of two shared-memory slots (/dev/shm).
child: python nearproc.py PREFIX DEVICE SEED W H fx fy cx cy"""
import os
import subprocess
import sys
import threading
import time

import numpy as np

CTL = 64       # float64: [0] request seq, [2:18] / [18:34] request c2w (by seq parity), [40] done seq, [41] done slot


def _maps(prefix, W, H, create):
    mode = "w+" if create else "r+"
    ctl = np.memmap(prefix + "ctl", dtype=np.float64, mode=mode, shape=(CTL,))
    slots = [(np.memmap(f"{prefix}img{i}", dtype=np.float16, mode=mode, shape=(H, W, 3)),
              np.memmap(f"{prefix}z{i}", dtype=np.float32, mode=mode, shape=(H, W)),
              np.memmap(f"{prefix}c2w{i}", dtype=np.float32, mode=mode, shape=(4, 4))) for i in range(2)]
    return ctl, slots


class NearTracer:
    """request(c2w) asks for a traced view (the latest request wins); latest() -> (seq, (img [H,W,3] f16, z [H,W], c2w))."""
    def __init__(self, dev, seed, K, W, H):
        self.prefix = f"/dev/shm/endless_near_{os.getpid()}_"
        self.ctl, self.slots = _maps(self.prefix, W, H, True)
        self.ctl[:] = 0
        self.n = 0
        args = [sys.executable, os.path.abspath(__file__), self.prefix, dev, str(seed), str(W), str(H),
                *(str(float(v)) for v in (K[0][0], K[1][1], K[0][2], K[1][2]))]
        self.p = subprocess.Popen(args, env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))

    def request(self, c2w):
        self.n += 1
        o = 2 + 16 * (self.n % 2)
        self.ctl[o:o + 16] = np.asarray(c2w, dtype=np.float64).reshape(-1)
        self.ctl[0] = self.n

    def latest(self):
        seq, s = int(self.ctl[40]), int(self.ctl[41])
        if seq == 0:
            return 0, None
        img, z, c2w = self.slots[s]
        return seq, (np.array(img), np.array(z), np.array(c2w))


def main():
    prefix, dev, seed, W, H = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    fx, fy, cx, cy = (float(v) for v in sys.argv[6:10])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    import torch
    import raytrace as RT
    torch.cuda.set_device(dev)
    torch.set_num_threads(2)
    ctl, slots = _maps(prefix, W, H, False)
    for f in os.listdir("/dev/shm"):          # both sides hold the mappings now; the names are not needed any more
        if ("/dev/shm/" + f).startswith(prefix):
            os.unlink("/dev/shm/" + f)
    parent = os.getppid()
    state = {"scene": None, "busy": False}

    def build(x, z):
        sc = RT.Scene(x - 130, x + 130, z - 130, z + 130, seed=seed, device=dev)
        sc.cx, sc.cz, sc.t_refresh = x, z, time.time()
        return sc

    def build_bg(x, z):
        try:
            sc = build(x, z)
            torch.cuda.synchronize()
            state["scene"] = sc
        except Exception as e:
            print(f"[near] {type(e).__name__}: {e}", flush=True)
        state["busy"] = False

    done = 0
    print(f"[near] tracer on {dev}", flush=True)
    while True:
        n = int(ctl[0])
        if n == done:
            if os.getppid() != parent:
                return
            time.sleep(0.001)
            continue
        done = n
        o = 2 + 16 * (n % 2)
        c2w = np.array(ctl[o:o + 16], dtype=np.float32).reshape(4, 4)
        x, z = float(c2w[0, 3]), float(c2w[2, 3])
        sc = state["scene"]
        if sc is None or abs(sc.cx - x) > 100 or abs(sc.cz - z) > 100:
            sc = state["scene"] = build(x, z)
        elif (abs(sc.cx - x) > 25 or abs(sc.cz - z) > 25) and not state["busy"]:
            state["busy"] = True
            threading.Thread(target=build_bg, args=(x, z), daemon=True).start()
        elif time.time() - sc.t_refresh > 2.0:
            sc.refresh_textures()
            sc.t_refresh = time.time()
        try:
            img, zz = RT.render(RT.Raytracer(sc, sky=False), c2w, K, W, H, haze=False, ss=1)
            s = (int(ctl[41]) + 1) % 2
            slots[s][0][:] = img.half().cpu().numpy()
            slots[s][1][:] = zz.cpu().numpy()
            slots[s][2][:] = c2w
            ctl[41] = s
            ctl[40] = n
        except Exception as e:
            print(f"[near] {type(e).__name__}: {e}", flush=True)
            time.sleep(1.0)


if __name__ == "__main__":
    main()
