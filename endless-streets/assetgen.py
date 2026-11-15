"""Image generator for the town: Qwen-Image-2.1-Turbo, FP8 blocks, on one 24 GB card.

Facades, cut-outs (trees, statues, benches, kiosks), seamless ground textures and posters. The walk servers POST
what they need, nearest first; the worker generates in that order and writes assets_cache/<version>_seed<seed>/<key>
(.jpg, or .png for cut-outs; atomic rename), so readers only ever see whole files.

usage: assetgen.sh [--port 8097] [--check JOBS.json]
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cityassets as CA  # noqa: E402

TURBO_DIR = os.path.expanduser("~/models/imggen/Qwen-Image-2.1-Turbo")


def _fp8(module):
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_
    from torchao.quantization.granularity import PerRow
    quantize_(module, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
              filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear))


def load():
    """BF16 on the CPU, then each block moved and quantized on its own, so the card never holds a BF16 copy
    of a whole model (DiT 14 GB + text encoder 17 GB BF16 would not fit in 24 GB). The vision tower stays on
    the CPU: text-to-image never calls it."""
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(TURBO_DIR, dtype=torch.bfloat16)
    dit = pipe.transformer
    for blk in dit.transformer_blocks:
        blk.to("cuda")
        _fp8(blk)
    dit.to("cuda")
    te = pipe.text_encoder
    lm = getattr(te.model, "language_model", te.model)
    for layer in lm.layers:
        layer.to("cuda")
        _fp8(layer)
    lm.to("cuda")
    te.lm_head.to("cuda")
    pipe.vae.to("cuda")
    torch.cuda.empty_cache()
    type(pipe)._execution_device = property(lambda self: torch.device("cuda"))
    dit.set_attention_backend("_native_cudnn")
    if os.environ.get("ASSETGEN_COMPILE", "1") == "1":
        _compile_blocks(dit)
        warm(pipe)
    return pipe


def _compile_blocks(dit):
    """Compile the transformer blocks only (as in imggen-moderation gen_worker): the top-level forward builds RoPE
    positions from Python lists and would recompile per prompt length; the prefill step (kv_cache_mode='extract')
    passes per-length ints, so it runs eagerly."""
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    for blk in dit.transformer_blocks:
        orig = blk.forward
        comp = torch.compile(orig, dynamic=None)

        def forward(*args, _orig=orig, _comp=comp, **kwargs):
            if kwargs.get("kv_cache_mode") == "extract":
                return _orig(*args, **kwargs)
            return _comp(*args, **kwargs)
        blk.forward = forward


def warm(pipe):
    """Every generation size once (twice for the first, so automatic-dynamic settles) before serving."""
    t = time.time()
    for k, (w, h) in enumerate([CA.BUCKETS[0]] + CA.BUCKETS):
        pipe(prompt="一栋楼房的正立面照片" + "，白墙" * k, width=w, height=h,
             generator=torch.Generator("cpu").manual_seed(0), output_type="pil")
    print(f"[assetgen] warm-up {time.time() - t:.0f} s", flush=True)


SEG_DIR = os.path.expanduser("~/models/segformer-b2-cityscapes")
NOT_FACADE = (0, 1, 9, 10)          # road, sidewalk, terrain, sky (cityscapes ids)
_seg = None


def crop_facade(img, dev="cuda"):
    """The model draws the whole building with sky, street and neighbours; keep only the front wall.
    rows: the span where most of the central columns are facade; columns: the run around the image centre
    whose roofline reaches the main roofline (neighbours are lower or absent)."""
    global _seg
    import numpy as np
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    if _seg is None:
        _seg = (SegformerImageProcessor.from_pretrained(SEG_DIR),
                SegformerForSemanticSegmentation.from_pretrained(SEG_DIR).to(dev).eval())
    proc, net = _seg
    with torch.no_grad():
        x = proc(images=img, return_tensors="pt")["pixel_values"].to(dev)
        lg = net(pixel_values=x).logits
        lab = torch.nn.functional.interpolate(lg, size=(img.height, img.width), mode="bilinear").argmax(1)[0]
    fac = ~torch.isin(lab, torch.tensor(NOT_FACADE, device=lab.device))
    fac = fac.cpu().numpy()
    H, W = fac.shape
    c0, c1 = int(W * 0.3), int(W * 0.7)
    rows = np.nonzero(fac[:, c0:c1].mean(1) > 0.6)[0]
    if len(rows) < H * 0.3:
        return img, None
    top, bot = int(rows[0]), int(rows[-1]) + 1
    probe = fac[min(top + max(3, (bot - top) // 30), bot - 1)]
    cols = probe.copy()
    l = r = W // 2
    while l > 0 and cols[l - 1]:
        l -= 1
    while r < W and cols[r]:
        r += 1
    colf = fac[top:bot].mean(0)
    while r - l > W * 0.3 and colf[l] < 0.9:      # side strips of sky / neighbour seen past the corner
        l += 1
    while r - l > W * 0.3 and colf[r - 1] < 0.9:
        r -= 1
    if r - l < W * 0.3:
        return img, None
    return img.crop((l, top, r, bot)), (l, top, r, bot)


def cutout(img):
    """RGBA cut-out cropped to its opaque part. If the model returned no alpha, white background -> transparent."""
    import numpy as np
    from PIL import Image
    a = np.asarray(img.convert("RGBA")).copy()
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if (border[:, 3] > 127).mean() > 0.5:
        # the model drew a background (white, sky) instead of transparency: key out the border colour.
        # Distance to the per-row border colour handles a sky gradient; keep the largest connected part.
        from scipy import ndimage
        rgb = a[..., :3].astype(np.float32)
        ref = np.median(np.stack([rgb[:, :4], rgb[:, -4:]], 1).reshape(rgb.shape[0], -1, 3), 1)   # [H,3]
        dist = np.abs(rgb - ref[:, None, :]).max(-1)
        fg = ndimage.binary_opening(dist > 28, iterations=1)
        lab, n = ndimage.label(fg)
        if n:
            big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
            fg = ndimage.binary_fill_holes(lab == big) | (fg & ndimage.binary_dilation(lab == big, iterations=6))
        a[..., 3] = np.where(fg, 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(a[..., 3] > 127)
    if len(ys) < 100:
        return Image.fromarray(a)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return Image.fromarray(a)


def seamless(img):
    """Square texture that repeats without a seam: blend the image with a copy shifted by half a period, the copy
    taking over near the borders (where it is continuous across the wrap) and the original in the middle."""
    import numpy as np
    from PIL import Image
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    n = min(a.shape[:2])
    a = a[(a.shape[0] - n) // 2:(a.shape[0] - n) // 2 + n, (a.shape[1] - n) // 2:(a.shape[1] - n) // 2 + n]
    b = np.roll(a, (n // 2, n // 2), (0, 1))
    r = np.abs(np.arange(n) + 0.5 - n / 2) / (n / 2)          # 0 centre .. 1 border
    w = np.clip((1 - np.maximum(r[:, None], r[None, :])) / 0.35, 0, 1)[..., None]
    w = w * w * (3 - 2 * w)
    return Image.fromarray((a * w + b * (1 - w)).clip(0, 255).astype(np.uint8))


def generate(pipe, job):
    out = pipe(prompt=job["prompt"], width=job["w"], height=job["h"],
               generator=torch.Generator("cpu").manual_seed(job["gseed"]), output_type="pil").images[0]
    kind = job.get("kind", "facade")
    if kind == "cutout":
        return cutout(out)
    if out.mode == "RGBA":
        from PIL import Image
        bg = Image.new("RGB", out.size, (255, 255, 255))
        bg.paste(out, mask=out.getchannel("A"))
        out = bg
    out = out.convert("RGB")
    if kind == "seamless":
        return seamless(out)
    if kind == "facade":
        out, box = crop_facade(out)
    return out


class Queue:
    """Per-client want lists (each client resends its full list, nearest first); the worker takes the head of
    each client's list in turn, so two walk servers share the card fairly."""
    def __init__(self):
        self.lists = {}
        self.cv = threading.Condition()
        self.done = 0
        self.gen_s = []
        self.turn = 0

    def put(self, client, seed, jobs):
        with self.cv:
            self.lists[client] = [dict(j, seed=seed) for j in jobs if not CA.have(seed, j["key"])]
            self.cv.notify()

    def take(self):
        with self.cv:
            while True:
                names = [c for c, l in self.lists.items() if l]
                if names:
                    self.turn += 1
                    c = names[self.turn % len(names)]
                    j = self.lists[c].pop(0)
                    if not CA.have(j["seed"], j["key"]):
                        return j
                    continue
                self.cv.wait()


def worker(pipe, q):
    while True:
        j = q.take()
        t = time.perf_counter()
        img = generate(pipe, j)
        dt = time.perf_counter() - t
        p = CA.path(j["seed"], j["key"])
        tmp = p.with_name("tmp_" + p.name)
        if p.suffix == ".png":
            img.save(tmp)
        else:
            img.save(tmp, quality=92)
        os.replace(tmp, p)
        (p.with_suffix(".txt")).write_text(j["prompt"])
        q.done += 1
        q.gen_s = (q.gen_s + [dt])[-50:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--check", default=None, help="generate one image to this path and exit")
    args = ap.parse_args()
    t = time.time()
    pipe = load()
    print(f"[assetgen] loaded in {time.time() - t:.0f} s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    if args.check:
        import json
        jobs = json.load(open(args.check))
        for j in jobs:
            for k in range(2):
                torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                img = generate(pipe, j)
                print(f"{j['key']} {j['w']}x{j['h']} {time.perf_counter() - t:.2f} s "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            img.save(j["out"], quality=92)
        return
    from fastapi import FastAPI
    import uvicorn
    q = Queue()
    threading.Thread(target=worker, args=(pipe, q), daemon=True).start()
    app = FastAPI()

    @app.post("/want")
    def want(req: dict):
        q.put(req["client"], int(req["seed"]), req["jobs"])
        return {"queued": sum(len(l) for l in q.lists.values())}

    @app.get("/stat")
    def stat():
        g = sorted(q.gen_s)
        return {"done": q.done, "queued": {c: len(l) for c, l in q.lists.items()},
                "gen_s_median": g[len(g) // 2] if g else None,
                "mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}

    print(f"[assetgen] http://127.0.0.1:{args.port}/", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
