#!/usr/bin/env python
"""The fast path for the WorldMirror forward, layered and individually refusable.

Tile content is a pure function of (seed, tile, spec, code); these layers are part of
"code", so stream_server hashes this module and the flag set into the world version.

  LAB_SKIP_HEADS=1  (default) drop the depth/points/normal heads at load. The tile
                    pipeline consumes splats, gs_depth(+conf) and camera_poses only;
                    those three heads were 25% of the forward, computed and discarded.
                    Numerically a no-op for what remains.
  LAB_ROPE_CACHE=1  (default) cache the RoPE sin/cos gather per token-grid shape.
                    The stock rope calls positions.max().item() twice per block --
                    ~96 CPU syncs per forward -- and rebuilds the same trig table
                    each time. The grid is a function of the shape alone here, so the
                    first call computes exactly what the stock code computes and later
                    calls replay it. Bitwise identical, and capture-safe.
  LAB_SYNC_FREE=1   (default) capture-safe conditioning: normalize_poses without the
                    NaN branches (the rig is sanitised upstream) and the camera
                    inverse in closed form instead of LU. The pose tokens shift by
                    float rounding; required for CUDA graphs.
  LAB_COMPILE=1     torch.compile the trunk blocks, patch embed and gs head with
                    dynamic shapes, so a changed view count does not recompile.
  LAB_CUDAGRAPH=1   whole-forward CUDA graphs, one per view count, lazily captured
                    into a shared memory pool. No padding: a padded view would take
                    part in global attention and change every real view's output.
"""
import os
from collections import OrderedDict

import torch

FLAGS = {
    "skip_heads": os.environ.get("LAB_SKIP_HEADS", "1") == "1",
    "rope_cache": os.environ.get("LAB_ROPE_CACHE", "1") == "1",
    "sync_free": os.environ.get("LAB_SYNC_FREE", "1") == "1",
    "compile": os.environ.get("LAB_COMPILE", "0") == "1",
    "cudagraph": os.environ.get("LAB_CUDAGRAPH", "0") == "1",
}


def stamp():
    """What the world version must include: the layers that can change tile bytes."""
    return {k: v for k, v in sorted(FLAGS.items())}


# ---------------------------------------------------------------- rope cache
def _patch_rope(rope):
    """Replace rope.forward with a shape-keyed replay of its own first result."""
    if rope is None or getattr(rope, "_lab_cached", False):
        return
    cache = OrderedDict()
    orig_get_sincos = rope._get_sincos_for_grid

    def forward(tokens, positions):
        B, _, N, C = tokens.shape
        key = (B, N, C, tokens.dtype)
        hit = cache.get(key)
        if hit is None:
            # stock computation, verbatim
            H = int(positions[..., 0].max().item() + 1)
            W = int(positions[..., 1].max().item() + 1)
            sin, cos = orig_get_sincos(H, W, tokens.device, tokens.dtype)
            idx = (positions[..., 0] * W + positions[..., 1]).long().view(-1)
            hit = (sin[idx].view(B, 1, N, C), cos[idx].view(B, 1, N, C))
            cache[key] = hit
            while len(cache) > 24:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        gsin, gcos = hit
        x1, x2 = tokens.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return tokens * gcos + rotated * gsin

    # Keep the cache lookup out of dynamo's sight: the cached tensors are shape-
    # specialised closure state, and tracing them recompiles per view count until
    # the recompile limit drops the whole attention frame back to eager. Disabled,
    # the blocks compile around three cheap eager kernels.
    rope.forward = torch._dynamo.disable(forward)
    rope._lab_cached = True


# ---------------------------------------------------------------- sync-free cond
def _normalize_poses_fast(extrinsics, padding=0.1, return_stats=False):
    """priors.normalize_poses for the finite, B=1 case, with no host syncs.

    Same operations in the same order as the stock path takes when every pose is
    finite and S > 10 -- which the rig guarantees -- minus the isnan/isinf branches
    that each cost a device round trip and forbid graph capture."""
    assert not return_stats
    B = extrinsics.shape[0]
    out = extrinsics.clone()
    for b in range(B):
        positions = extrinsics[b, :, :3, 3]
        if positions.shape[0] > 10:
            min_pos = torch.quantile(positions, 0.05, dim=0)
            max_pos = torch.quantile(positions, 0.95, dim=0)
        else:
            min_pos = torch.min(positions, dim=0)[0]
            max_pos = torch.max(positions, dim=0)[0]
        pos_range = max_pos - min_pos
        eps = torch.maximum(
            torch.full((), 1e-6, device=extrinsics.device),
            torch.abs(max_pos) * 1e-6)
        pos_range = torch.maximum(pos_range, eps)
        scale = torch.clamp(torch.max(pos_range), min=1e-6, max=1e6)
        center = (min_pos + max_pos) / 2.0
        actual = scale / (1 - 2 * padding)
        out[b, :, :3, 3] = torch.clamp((positions - center) / actual + 0.5, 0.0, 1.0)
    return out


def _patch_sync_free(inner):
    """Pose normalisation without device round trips; camera inverse in closed form."""
    import sys
    wm = sys.modules[type(inner).__module__]
    wm.normalize_poses = _normalize_poses_fast

    # rotmat_to_quat builds its 0.1 floor with torch.tensor(0.1).to(device) -- a
    # synchronizing host copy on every call, and a capture blocker. Same function,
    # floor built on device.
    from hyworld2.worldrecon.hyworldmirror.models.utils import rotation as _rot
    from hyworld2.worldrecon.hyworldmirror.models.utils import camera_utils as _cu
    _orig_tensor = torch.tensor

    def _rotmat_to_quat_fast(matrix):
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1)
        q_abs = _rot._sqrt_positive_part(torch.stack(
            [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
             1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1))
        quat_by_rijk = torch.stack([
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ], dim=-2)
        flr = q_abs.new_full((), 0.1)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
        # stock code picks the best-conditioned row by boolean mask, which calls
        # nonzero and syncs; the row index is just argmax, so gather it instead
        idx = q_abs.argmax(dim=-1)
        out = torch.gather(
            quat_candidates, -2,
            idx[..., None, None].expand(batch_dim + (1, 4))).squeeze(-2)
        out = torch.cat([out[..., 1:], out[..., :1]], dim=-1)  # rijk -> ijkr
        return _rot.standardize_quaternion(out)

    # keep the tail of the stock implementation authoritative: reuse it verbatim if
    # the shape of the function ever changes upstream, fall back to the original
    try:
        _rotmat_to_quat_fast(torch.eye(3, device=inner.cam_token.device
                                       if hasattr(inner, "cam_token")
                                       else "cuda").unsqueeze(0))
        _rot.rotmat_to_quat = _rotmat_to_quat_fast
        _cu.rotmat_to_quat = _rotmat_to_quat_fast
    except Exception:
        pass

    from hyworld2.worldrecon.hyworldmirror.models.utils.geometry import (
        closed_form_inverse_se3)
    from hyworld2.worldrecon.hyworldmirror.models.utils.camera_utils import (
        vector_to_camera_matrices)

    _homo = {}

    def transform_camera_vector(camera_params, h, w):
        ext, intr = vector_to_camera_matrices(camera_params, image_hw=(h, w))
        B, S = ext.shape[0], ext.shape[1]
        # torch.tensor(list, device=cuda) is a synchronizing copy; build the constant
        # on-device once per dtype (warmup fills the cache before any graph capture)
        row = _homo.get(ext.dtype)
        if row is None:
            row = torch.zeros(4, device=ext.device, dtype=ext.dtype)
            row[3] = 1.0
            _homo[ext.dtype] = row
        homo = row.view(1, 1, 1, 4).expand(B, S, 1, 4)
        w2c = torch.cat([ext, homo], dim=2)
        # R is orthonormal by construction (it comes off a unit quaternion), so the
        # inverse is [R^T | -R^T t] exactly; LU adds nothing but capture problems.
        c2w = closed_form_inverse_se3(w2c.reshape(B * S, 4, 4)).reshape(B, S, 4, 4)
        return c2w, intr

    inner.transform_camera_vector = transform_camera_vector


# ---------------------------------------------------------------- compile
def _compile(inner):
    """Fuse where the time is: the 48 trunk blocks, the DINOv2 patch embed, and the
    surviving DPT head. dynamic=True from the start, so the view count changing
    (frame blocks see it as batch, global blocks as sequence) never recompiles."""
    vgt = inner.visual_geometry_transformer
    for blocks in (vgt.frame_blocks, vgt.global_blocks):
        for i in range(len(blocks)):
            blocks[i] = torch.compile(blocks[i], dynamic=True)
    vgt.patch_embed = torch.compile(vgt.patch_embed, dynamic=True)
    if hasattr(inner, "gs_head"):
        inner.gs_head = torch.compile(inner.gs_head, dynamic=True)
    if hasattr(inner, "gs_renderer"):
        inner.gs_renderer.gs_head = torch.compile(inner.gs_renderer.gs_head,
                                                  dynamic=True)


# ---------------------------------------------------------------- cuda graphs
def _clone_tree(x):
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, dict):
        return {k: _clone_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_clone_tree(v) for v in x)
    return x


class GraphedForward:
    """model(views, cond_flags, is_inference=True), captured once per view count.

    The view count varies with the maze (corner cameras exist where walls do), so
    graphs are keyed by it and captured lazily on first sight, all drawing from one
    shared memory pool -- only one graph is ever mid-flight, the build lock upstream
    guarantees that. Inputs are copied into per-shape static buffers; outputs are
    cloned out so the next replay cannot overwrite a tile mid-postprocess.
    """

    MAX_GRAPHS = 12

    def __init__(self, model, use_amp):
        self.model = model
        self.use_amp = use_amp
        self.entries = OrderedDict()          # S -> (graph, static_in, static_out)
        self.pool = None

    def _run(self, views, cond_flags):
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.bfloat16):
            return self.model(views=views, cond_flags=cond_flags, is_inference=True)

    def __call__(self, views, cond_flags):
        S = views["img"].shape[1]
        key = (S, tuple(cond_flags))
        ent = self.entries.get(key)
        if ent is None:
            ent = self._capture(views, cond_flags)
            self.entries[key] = ent
            while len(self.entries) > self.MAX_GRAPHS:
                self.entries.popitem(last=False)
                torch.cuda.synchronize()
        else:
            self.entries.move_to_end(key)
        graph, static_in, static_out = ent
        for k, v in views.items():
            static_in[k].copy_(v)
        graph.replay()
        return _clone_tree(static_out)

    def _capture(self, views, cond_flags):
        static_in = {k: v.clone() for k, v in views.items()}
        # warm up on a side stream: cuBLAS/autotune handshakes must happen outside
        # capture, and every lazy init must have fired
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                out = self._run(static_in, cond_flags)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        del out
        graph = torch.cuda.CUDAGraph()
        ctx = {"pool": self.pool} if self.pool is not None else {}
        with torch.cuda.graph(graph, **ctx):
            static_out = self._run(static_in, cond_flags)
        if self.pool is None:
            self.pool = graph.pool()
        return graph, static_in, static_out


# ---------------------------------------------------------------- entry points
def disabled_heads():
    return ["depth", "points", "normal"] if FLAGS["skip_heads"] else None


_graphed = None


def prepare(pipe):
    """Apply the enabled layers to a freshly loaded pipeline. Idempotent."""
    global _graphed
    inner = pipe.model.module if hasattr(pipe.model, "module") else pipe.model
    if FLAGS["rope_cache"]:
        _patch_rope(getattr(inner.visual_geometry_transformer, "rope", None))
    if FLAGS["sync_free"]:
        _patch_sync_free(inner)
    if FLAGS["compile"]:
        _compile(inner)
    if FLAGS["cudagraph"]:
        if not FLAGS["sync_free"] or not FLAGS["rope_cache"]:
            raise RuntimeError("LAB_CUDAGRAPH needs LAB_ROPE_CACHE and LAB_SYNC_FREE")
        use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        _graphed = GraphedForward(pipe.model, use_amp)
    print(f"[fastinfer] {stamp()}", flush=True)


def run(pipe, views, cond_flags):
    """The forward pass recon.infer delegates to."""
    if _graphed is not None:
        return _graphed(views, cond_flags)
    inner = pipe.model.module if hasattr(pipe.model, "module") else pipe.model
    model_bf16 = getattr(inner, "enable_bf16", False)
    use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    with torch.amp.autocast("cuda", enabled=(not model_bf16 and use_amp),
                            dtype=torch.bfloat16):
        return pipe.model(views=views, cond_flags=cond_flags, is_inference=True)
