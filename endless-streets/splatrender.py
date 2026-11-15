"""Render tile splats with gsplat: colour, expected depth and alpha at an OpenCV view matrix."""
import torch
from gsplat import rasterization


def cat_tiles(tiles):
    """tiles: list of (splat dict in tile frame, origin xyz) -> one splat set in world frame."""
    ks = ("means", "scales", "quats", "rgb", "opacity")
    out = {k: [] for k in ks}
    for g, o in tiles:
        for k in ks:
            v = g[k]
            if k == "means":
                v = v + torch.as_tensor(o, device=v.device, dtype=v.dtype)
            out[k].append(v)
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def render(g, viewmat, K, W, H, alpha_threshold=0.05):
    """-> image [H,W,3] (not composited), expected z-depth [H,W] (0 where alpha <= threshold), alpha [H,W].
    Two passes because the gsplat build here is compiled for 3 channels only."""
    img, alpha, _ = rasterization(g["means"], g["quats"], g["scales"], g["opacity"], g["rgb"],
                                  viewmat[None], K[None], W, H, packed=False)
    z = (g["means"] @ viewmat[:3, :3].T + viewmat[:3, 3])[:, 2:3].expand(-1, 3).contiguous()
    wz, _, _ = rasterization(g["means"], g["quats"], g["scales"], g["opacity"], z,
                             viewmat[None], K[None], W, H, packed=False)
    a = alpha[0, ..., 0]
    depth = torch.where(a > alpha_threshold, wz[0, ..., 0] / a.clamp_min(1e-4), torch.zeros_like(a))
    return img[0].clamp(0, 1), depth, a
