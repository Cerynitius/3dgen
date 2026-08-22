#!/usr/bin/env python
"""WorldMirror feed-forward reconstruction, and an objective way to score it.

The offline worldgen path (WorldNav -> WorldStereo -> gen_gs_data -> 3DGS trainer)
costs ~41 min per world, of which WorldStereo alone is ~23 min. That can seed a world;
it cannot be what a player walks into. WorldMirror is a single feed-forward pass that
emits Gaussians directly, so it is the only thing here fast enough to be endless.

Everything the official CLI does around the forward pass is skipped -- no sky/edge
mask, no depth/normal/points PLY, no point compression -- because none of it survives
into the splats a renderer needs.

Scoring: the raycaster IS the ground truth. Any tile can be re-rendered from a held-out
camera with gsplat and compared against the raycast of the same pose, so a config is
chosen on measured PSNR rather than on which screenshot looked better.
"""
import os
import sys
import threading
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
for _v in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
           "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(_v, None)

sys.path.insert(0, "/home/zrx/HY-World-2.0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import tiles as T

WEIGHTS = "/home/zrx/HY-World-2.0/weights"
_pipeline = None
_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
        t0 = time.perf_counter()
        _pipeline = WorldMirrorPipeline.from_pretrained(
            WEIGHTS, subfolder="HY-WorldMirror-2.0")
        print(f"[recon] WorldMirror resident in {time.perf_counter() - t0:.1f}s",
              flush=True)
    return _pipeline


def infer(batch, extr, intr, use_priors=True):
    """The forward pass of pipeline._run_inference without the file plumbing.

    Feeding the true camera poses matters for more than alignment: unconditioned, the
    model reconstructs in its own normalised frame, so the per-splat scales come out far
    too small for the space they are stretched into and the field renders see-through.
    """
    pipe = get_pipeline()
    device = pipe.device
    # a handful of NaN pixels in one view (render glitch at grazing rays) poison the
    # ENTIRE forward -- every mean comes back NaN and the alignment SVD dies on it.
    # 10 pixels out of 11M are noise either way; neutralise them instead.
    batch = torch.nan_to_num(batch, nan=0.0, posinf=1.0, neginf=0.0)
    views = {"img": batch.to(device)}
    cond_flags = [0, 0, 0]
    if use_priors:
        e = extr.to(device)
        first = e[0, 0]
        e = torch.linalg.inv(first.float()).to(first.dtype).unsqueeze(0).unsqueeze(0) @ e
        views["camera_poses"] = e
        views["camera_intrs"] = intr.to(device)
        cond_flags[0] = 1
        cond_flags[2] = 1
    inner = pipe.model.module if hasattr(pipe.model, "module") else pipe.model
    model_bf16 = getattr(inner, "enable_bf16", False)
    use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(not model_bf16 and use_amp),
                                dtype=torch.bfloat16):
            pred = pipe.model(views=views, cond_flags=cond_flags, is_inference=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return pred, time.perf_counter() - t0


def build_tile(maze, tc, tr, spec, use_priors=True):
    """One tile end to end: rig -> forward pass -> metric frame -> crop -> prune."""
    t = {}
    t0 = time.perf_counter()
    batch, extr, intr, win, first_c2w, depths = T.render_rig(maze, tc, tr, spec)
    t["views"] = time.perf_counter() - t0

    with _lock:
        pred, fwd_s = infer(batch, extr, intr, use_priors)
    t["forward"] = fwd_s

    t1 = time.perf_counter()
    g, meta = T.postprocess(pred, spec, extr, intr, depths, maze, tc, tr)
    t["post"] = time.perf_counter() - t1
    t["total"] = time.perf_counter() - t0
    meta["timing"] = {k: round(v, 4) for k, v in t.items()}
    meta["tile"] = [tc, tr]
    meta["views"] = int(batch.shape[1])
    return g, meta, win


# ---------------------------------------------------------------- scoring
class SplatConvention:
    """How the model's raw splat tensors map onto what a rasteriser expects.

    The heads emit quaternions, colours and opacities in a convention the official
    exporter knows and a bare forward pass does not: quats may be wxyz or xyzw, colours
    may be SH DC coefficients or plain RGB, opacities may be logits or probabilities.
    Guessing wrong does not crash -- it quietly costs PSNR, which would then be blamed
    on the configuration being swept. So it is measured once, up front.
    """

    def __init__(self, quat="wxyz", color="sh", opacity="raw", cam="none"):
        self.quat, self.color, self.opacity, self.cam = quat, color, opacity, cam

    def apply(self, g):
        means, scales, quats, colors, opac = g
        if self.quat == "xyzw":
            quats = quats[:, [3, 0, 1, 2]]
        quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if self.color == "sh":
            colors = (0.28209479177387814 * colors + 0.5)
        colors = colors.clamp(0, 1)
        if self.opacity == "sigmoid":
            opac = torch.sigmoid(opac)
        return means, scales, quats, colors, opac.reshape(-1).clamp(0, 1)

    def viewmat(self, c2w, device):
        """The rig now builds cameras in OpenCV convention already (render.camera_basis),
        so the default is no flip at all -- the same inv(camtoworld) the official
        rasteriser applies. The flips stay reachable only so calibration can show that
        the alternatives are worse rather than assert it."""
        m = torch.as_tensor(c2w, dtype=torch.float32, device=device).clone()
        if "Y" in self.cam:
            m[:3, 1] *= -1
        if "Z" in self.cam:
            m[:3, 2] *= -1
        return torch.linalg.inv(m)

    def __repr__(self):
        return (f"cam={self.cam:6s} quat={self.quat} color={self.color:3s} "
                f"opacity={self.opacity}")


def rasterize(g, c2w, K, width, height, conv):
    """Render splats with gsplat, in the tile's own metric frame."""
    from gsplat import rasterization
    means, scales, quats, colors, opac = conv.apply(g)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=means.device)
    img, _, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
        viewmats=conv.viewmat(c2w, means.device).unsqueeze(0), Ks=Kt.unsqueeze(0),
        width=width, height=height, render_mode="RGB")
    return img[0].clamp(0, 1)              # [H,W,3]


def rasterize_depth(g, c2w, K, width, height, conv):
    """Expected depth from gsplat, in the same metres as the raycaster."""
    from gsplat import rasterization
    means, scales, quats, colors, opac = conv.apply(g)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=means.device)
    out, alpha, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
        viewmats=conv.viewmat(c2w, means.device).unsqueeze(0), Ks=Kt.unsqueeze(0),
        width=width, height=height, render_mode="RGB+ED")
    return out[0, ..., 3], alpha[0, ..., 0]        # depth [H,W], alpha [H,W]


def depth_metrics(g, cam, spec, conv, tol=0.10):
    """Geometry agreement, which darkness cannot flatter.

    Reported on pixels where BOTH the raycaster and the splats have something: a tile
    that renders nothing would otherwise score a perfect error on an empty set.
    """
    W, H = spec.wh()
    fx = (W / 2) / np.tan(spec.fov_x / 2)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=np.float32)
    d_pred, alpha = rasterize_depth(g, cam["c2w"], K, W, H, conv)
    d_gt = cam["depth"]
    valid = (d_gt > 1e-3) & (alpha > 0.5)
    frac = float(valid.float().mean())
    if frac < 1e-4:
        return {"depth_mae": float("nan"), "depth_within": 0.0, "filled": frac}
    err = (d_pred[valid] - d_gt[valid]).abs()
    return {"depth_mae": float(err.median()),
            "depth_within": float((err < tol).float().mean()),
            "filled": frac}


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def score_tile(g, cams, spec, conv):
    """Mean PSNR of the reconstruction against the raycast, over held-out cameras."""
    W, H = spec.wh()
    fx = (W / 2) / np.tan(spec.fov_x / 2)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=np.float32)
    out = []
    for cam in cams:
        img = rasterize(g, cam["c2w"], K, W, H, conv)
        gt = cam["gt"].permute(1, 2, 0)
        out.append(psnr(img, gt))
    return float(np.mean(out)), out


def calibrate_convention(g, cams, spec, verbose=True):
    """Settle the splat convention by measurement, not by reading the exporter.

    Eight combinations, one held-out camera each; the winner is whichever actually
    reconstructs the raycast. Reported with the runner-up so a near-tie is visible
    rather than hidden behind a single number.
    """
    results = []
    for cam in ("flipY", "none", "flipZ", "flipYZ"):
        for quat in ("wxyz", "xyzw"):
            for color in ("sh", "raw"):
                for opacity in ("sigmoid", "raw"):
                    conv = SplatConvention(quat, color, opacity, cam)
                    try:
                        m, _ = score_tile(g, cams[:2], spec, conv)
                        cov = _coverage(g, cams[0], spec, conv)
                    except Exception as e:
                        m, cov = float("nan"), 0.0
                        if verbose:
                            print(f"    {conv}  failed: {str(e)[:60]}")
                    results.append((m, cov, conv))
    results = [r for r in results if not np.isnan(r[0])]
    # Rank on PSNR, but carry coverage so an empty render cannot win by default: a
    # blank frame scores ~21 dB against a mostly dark labyrinth, which is why the first
    # calibration reported all eight combinations as identical.
    results.sort(key=lambda x: -x[0])
    if verbose:
        for m, cov, conv in results[:8]:
            print(f"    {m:6.2f} dB  coverage {cov * 100:5.1f}%   {conv}")
    live = [r for r in results if r[1] > 0.02]
    if not live:
        raise RuntimeError("every convention rendered an empty frame -- "
                           "the tile is not where the cameras are looking")
    return live[0][2], results


def _coverage(g, cam, spec, conv):
    """Fraction of the frame the splats actually paint."""
    W, H = spec.wh()
    fx = (W / 2) / np.tan(spec.fov_x / 2)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=np.float32)
    img = rasterize(g, cam["c2w"], K, W, H, conv)
    return float((img.sum(-1) > 0.01).float().mean())
