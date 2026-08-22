#!/usr/bin/env python
"""Raycast the infinite labyrinth through a sliding window.

The renderer cannot hold an infinite occupancy grid, so it walks a window: the cell
box around whatever is being looked at, rasterised on demand and thrown away. Two
things have to survive that, or the world stops being one world:

  textures  are keyed on WORLD position, never on the window. Keying them on the
            local array makes the brick courses slide under the player whenever the
            window re-centres, and puts a different mortar pattern on either side of
            every tile seam.
  skirt     the window's outer ring stays solid. A ray that escapes the array hits
            nothing, and nothing has no surface to shade -- it comes back as a black
            hole in the middle of a corridor.

Shading is torchlight only -- point lights with shadow rays, no sun and no sky --
because the reconstruction model rebuilds whatever the views show it, and a seed lit
from an open sky can only ever reconstruct an open-air maze.
"""
import math

import numpy as np
import torch

from infinite_maze import CELL_M, WALL_H, EYE_H, DIRS

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- lighting, tuned by measurement (see bench_light.py)
#
# These were leaving the world at a median pixel of 0.09 with nothing above 0.21: legible
# as an image, unreadable as a place to walk. The cause was not the renderer -- the
# reconstruction tracks the raycaster to 1.02x -- but the lighting itself. Torches sat
# 2.5 cells apart, which is 10 m, so with pure inverse-square the nearest one was already
# down to 0.04 before shadowing took another 2.5x.
#
# The fix is more torches and a gentler near-field falloff, not more ambient: flat
# ambient lifts the mean but flattens exactly the shading that makes a corridor readable.
REACH_M = 22.0               # torchlight carries this far before the taper bites
INTENSITY = 46.0             # radiance at the flame
FALLOFF_K = 0.45             # d^2 coefficient; smaller = light travels further
AMBIENT = 0.16   # variant H1               # only reaches rays that hit NOTHING -- see below
# FILL is the real ambient floor. AMBIENT looks like it should be, and is not: the lit
# branch overwrites col wholesale for every pixel with a surface behind it, which in a
# closed vault is all of them, so AMBIENT is reachable only by a ray that escapes. Raising
# it from 0.10 to 0.45 changed nothing at all, which is how this surfaced.
#
# 0.15 swept against the dark views specifically: it lifts the darkest view from 0.092 to
# 0.235 so no corridor is unnavigable, while keeping p90/p10 contrast at 3.6x so torchlight
# still reads as torchlight rather than as flat fill.
FILL = 0.28   # variant H1: brighter reconstruction views; the GAME stays gloomy, the MODEL needs signal

# Denser torches made the window's light list four times longer, and two terms in shade()
# are O(pixels x lights): the distance matrix and, far worse, the flame-visibility test,
# which materialises (pixels, lights, 3) -- 383 MB per call at 168 torches. Tile build
# went 1.8s -> 8.7s, which is the difference between outrunning a walker and not.
# Every ray in a view shares one origin, so the lights can be culled against that origin
# once instead of per pixel. The caps are far above the 10 that actually get sampled.
# Tested and rejected: giving the ceiling ribs to lock depth onto changed its holes
# not at all (25.5% -> 25.6%) and cost depth (14.4 -> 15.4 cm). The ceiling is not
# unresolved for want of features.
CEIL_RIBS = False
LIGHT_CAP = 48               # nearest torches considered for shading
FLAME_CAP = 24               # nearest torches drawn as visible flames


# ---------------------------------------------------------------- noise
def _hash2(a, b):
    return torch.frac(torch.sin(a * 127.1 + b * 311.7) * 43758.5453)


def _vnoise(x, y):
    xi, yi = torch.floor(x), torch.floor(y)
    xf, yf = x - xi, y - yi
    u = xf * xf * (3 - 2 * xf)
    v = yf * yf * (3 - 2 * yf)
    a, b = _hash2(xi, yi), _hash2(xi + 1, yi)
    c, d = _hash2(xi, yi + 1), _hash2(xi + 1, yi + 1)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v


def _fbm(x, y, octaves=4):
    out, amp, freq = torch.zeros_like(x), 0.5, 1.0
    for _ in range(octaves):
        out = out + amp * _vnoise(x * freq, y * freq)
        amp, freq = amp * 0.5, freq * 2.1
    return out


# ---------------------------------------------------------------- window
class Window:
    """A cell box of the infinite maze, resident on the GPU.

    `off` is the window's origin in world GRID units, which is what turns a local
    hit position back into the world position the textures are keyed on."""

    __slots__ = ("occ_t", "off_x", "off_z", "lights", "box", "n_torch")

    def __init__(self, occ_t, off_x, off_z, lights, box):
        self.occ_t = occ_t
        self.off_x, self.off_z = off_x, off_z
        self.lights = lights
        self.box = box
        self.n_torch = int(lights.shape[0])

    def to_local(self, pos_world_grid):
        x, y, z = pos_world_grid
        return (x - self.off_x, y, z - self.off_z)


def build_window(maze, c0, r0, cols, rows, device=DEV, light_halo=6):
    """Rasterise the cell box and gather the torches that light it.

    Torches are collected from a halo beyond the box: a torch just outside the window
    still throws light and shadow into it, and dropping it would draw the window
    boundary on the floor as a lighting discontinuity."""
    occ = maze.occupancy(c0, r0, cols, rows)
    occ_t = torch.from_numpy(occ).to(device)
    t = maze.torches_in(c0 - light_halo, r0 - light_halo,
                        cols + 2 * light_halo, rows + 2 * light_halo)
    off_x, off_z = 2.0 * c0, 2.0 * r0
    if t:
        lights = torch.tensor([[p[0] - off_x, p[1], p[2] - off_z] for p in t],
                              device=device, dtype=torch.float32)
    else:
        lights = torch.zeros(0, 3, device=device)
    return Window(occ_t, off_x, off_z, lights, (c0, r0, cols, rows))


# ---------------------------------------------------------------- raycast
def dda(o, d, occ_t, k_steps=220, t_far=400.0, ceiling=WALL_H):
    """Walk the grid. Returns t_hit, kind (0 miss, 1 wall, 2 floor, 3 ceiling), normal."""
    n = o.shape[0]
    gz, gx = occ_t.shape
    d = torch.where(d.abs() < 1e-8, torch.full_like(d, 1e-8), d)
    ox, oy, oz = o[:, 0], o[:, 1], o[:, 2]
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]

    ix = torch.floor(ox).long()
    iz = torch.floor(oz).long()
    step_x = torch.sign(dx).long()
    step_z = torch.sign(dz).long()
    t_dx, t_dz = (1.0 / dx).abs(), (1.0 / dz).abs()
    t_mx = ((ix + (step_x > 0).long()).float() - ox) / dx
    t_mz = ((iz + (step_z > 0).long()).float() - oz) / dz

    t_hit = torch.full((n,), float("inf"), device=o.device)
    kind = torch.zeros(n, dtype=torch.long, device=o.device)
    normal = torch.zeros(n, 3, device=o.device)
    t_cur = torch.zeros(n, device=o.device)

    # the vault is closed: floor and ceiling are always reached eventually
    t_floor = torch.where(dy < 0, -oy / dy, torch.full_like(oy, float("inf")))
    t_ceil = torch.where(dy > 0, (ceiling - oy) / dy, torch.full_like(oy, float("inf")))

    for _ in range(k_steps):
        active = (kind == 0) & (t_cur < t_far)
        if not active.any():
            break
        t_next = torch.minimum(t_mx, t_mz)

        for t_plane, k, nrm in ((t_floor, 2, (0.0, 1.0, 0.0)),
                                (t_ceil, 3, (0.0, -1.0, 0.0))):
            hit = active & (t_plane <= t_next) & (t_plane >= t_cur - 1e-6)
            t_hit = torch.where(hit, t_plane, t_hit)
            kind = torch.where(hit, torch.full_like(kind, k), kind)
            normal[hit] = torch.tensor(nrm, device=o.device)
            active = active & ~hit

        take_x = t_mx <= t_mz
        ix = torch.where(active & take_x, ix + step_x, ix)
        iz = torch.where(active & ~take_x, iz + step_z, iz)
        t_cur = torch.where(active, t_next, t_cur)
        t_mx = torch.where(active & take_x, t_mx + t_dx, t_mx)
        t_mz = torch.where(active & ~take_x, t_mz + t_dz, t_mz)

        inb = (ix >= 0) & (ix < gx) & (iz >= 0) & (iz < gz)
        cell = occ_t[iz.clamp(0, gz - 1), ix.clamp(0, gx - 1)] & inb & active
        y_ent = oy + dy * t_cur
        face = cell & (y_ent >= 0) & (y_ent <= ceiling)
        t_hit = torch.where(face, t_cur, t_hit)
        kind = torch.where(face, torch.ones_like(kind), kind)
        zero = torch.zeros(n, device=o.device)
        nrm_x = torch.stack([-step_x.float(), zero, zero], 1)
        nrm_z = torch.stack([zero, zero, -step_z.float()], 1)
        normal = torch.where((face & take_x).unsqueeze(1), nrm_x, normal)
        normal = torch.where((face & ~take_x).unsqueeze(1), nrm_z, normal)

    return t_hit, kind, normal


def shade(o, d, win, n_lights=10):
    """Torchlight only. `o`/`d` are in WINDOW-local grid units; textures are looked
    up at the world position, which is the local hit plus the window offset."""
    occ_t, lights = win.occ_t, win.lights
    n = o.shape[0]
    # Every ray in one view shares an origin, so cull the light list against it once.
    # Without this the cost of a tile scales with how many torches happen to be in the
    # window rather than with how many can affect the image.
    if lights.shape[0] > LIGHT_CAP:
        d0 = (lights - o[0].unsqueeze(0)).pow(2).sum(-1)
        lights = lights[torch.topk(d0, LIGHT_CAP, largest=False).indices]
    t, kind, nrm = dda(o, d, occ_t)
    t = torch.where(torch.isinf(t), torch.full_like(t, 1e3), t)
    p = o + d * t.unsqueeze(1)

    # world position, in metres -- the frame every texture is keyed on
    px = (p[:, 0] + win.off_x) * CELL_M
    py = p[:, 1] * CELL_M
    pz = (p[:, 2] + win.off_z) * CELL_M

    # ---- albedo: cave-hewn rock. Low-frequency by design: the observation pipeline
    # cannot carry 10 cm brickwork at grazing angles, so the texture budget is spent
    # on metre-scale features that survive everywhere.
    alb = torch.zeros(n, 3, device=o.device)
    wall = kind == 1
    u = torch.where(nrm[:, 0].abs() > 0.5, pz, px)
    v = py
    # rough rock: two fbm scales, large tone swings
    rockn = _fbm(u * 2.0, v * 2.0, 4)
    macro = _fbm(u * 0.6, v * 0.6, 3)
    rock = torch.stack([0.46 + 0.10 * macro, 0.40 + 0.09 * macro, 0.33 + 0.08 * macro], 1)
    rock = rock * (0.58 + 0.75 * rockn.unsqueeze(1))
    # chisel courses: irregular horizontal grooves from the digging
    course = torch.floor(v / 0.55 + _hash2(torch.floor(u / 1.4), torch.zeros_like(v)) * 0.7)
    groove = (torch.frac(v / 0.55 + _hash2(course, torch.zeros_like(v)) * 0.4) < 0.06)
    rock = torch.where(groove.unsqueeze(1), rock * 0.60, rock)
    alb = torch.where(wall.unsqueeze(1), rock, alb)

    floor = kind == 2
    # worn bedrock floor: broad slabs (2.2 m) with soft seams, sand pooling in hollows
    tx, tz = torch.floor(px / 2.2), torch.floor(pz / 2.2)
    fj = _hash2(tx, tz)
    seam = (torch.frac(px / 2.2) < 0.03) | (torch.frac(pz / 2.2) < 0.03)
    slab = torch.stack([0.50 + 0.12 * fj, 0.44 + 0.10 * fj, 0.36 + 0.08 * fj], 1)
    wear = _fbm(px * 1.2, pz * 1.2, 4).unsqueeze(1)          # large-scale wear variation
    slab = slab * (0.78 + 0.40 * wear)
    sand = _fbm(px * 0.5 + 31.7, pz * 0.5 + 17.3, 3).unsqueeze(1)
    sandy = torch.tensor([0.62, 0.54, 0.40], device=o.device).unsqueeze(0)
    slab = torch.where(sand > 0.62, sandy * (0.85 + 0.3 * wear), slab)   # sand drifts
    slab = torch.where(seam.unsqueeze(1), slab * 0.72, slab)
    alb = torch.where(floor.unsqueeze(1), slab, alb)

    ceil = kind == 3
    vaultn = _fbm(px * 1.5, pz * 1.5, 4).unsqueeze(1)
    vault = torch.tensor([0.34, 0.30, 0.26], device=o.device).unsqueeze(0) * (0.65 + 0.55 * vaultn)
    if CEIL_RIBS:
        rib = ((torch.frac(px / 1.6) < 0.05) | (torch.frac(pz / 1.6) < 0.05)).unsqueeze(1)
        vault = torch.where(rib, vault * 0.55, vault)
    alb = torch.where(ceil.unsqueeze(1), vault, alb)

    # A ray that hit nothing would otherwise shade to pure black and read as a hole.
    # The skirt makes this near-impossible, but a grazing ray can still slip out.
    miss = kind == 0
    alb = torch.where(miss.unsqueeze(1),
                      torch.full((n, 3), 0.22, device=o.device), alb)

    # ---- torchlight
    col = alb * AMBIENT                  # dim, not black: pure black reconstructs as noise
    # variant L (cave): bright diffuse base so nothing is ever dark, plus braziers --
    # the flame sits LOW (0.55 m) and shines upward: floors catch the pool directly,
    # walls get lit from below, ceilings stay in shadow. The base term keeps every
    # surface above the model's signal floor, so there are no torch pockets.
    flat = torch.full((n, 1), 0.45, device=o.device)
    flat = torch.where(floor.unsqueeze(1), torch.full((n, 1), 1.05, device=o.device), flat)
    flat = torch.where(ceil.unsqueeze(1), torch.full((n, 1), 0.40, device=o.device), flat)
    out = alb * flat * 0.95
    if lights.shape[0]:
        braz = lights.clone()
        braz[:, 1] = 0.55 / CELL_M                        # brazier bowl height, grid units
        dist_all = (braz.unsqueeze(0) - p.unsqueeze(1)).pow(2).sum(-1).sqrt()
        kk = min(8, braz.shape[0])
        near_d, near_i = torch.topk(dist_all, kk, dim=1, largest=False)
        acc = torch.zeros(n, 3, device=o.device)
        for j in range(kk):
            lp = braz[near_i[:, j]]
            dist = near_d[:, j].clamp(min=1e-3)
            ldir = (lp - p) / dist.unsqueeze(1)
            ndl = (nrm * ldir).sum(1).clamp(min=0.0)
            d_m = dist * CELL_M
            falloff = 1.0 / (1.0 + (d_m * 0.5) ** 2)
            window_ = torch.exp(-(d_m / 14.0) ** 2)
            wallpool = torch.exp(-((py - 0.55) / 1.4) ** 2).unsqueeze(1)  # hug the floor line on walls
            boost = torch.where(wall.unsqueeze(1), wallpool, torch.ones(n, 1, device=o.device))
            acc = acc + (ndl * falloff * window_).unsqueeze(1) * boost * \
                torch.tensor([1.0, 0.62, 0.30], device=o.device).unsqueeze(0)
        out = out + alb * acc * 4.2
    out = torch.where(miss.unsqueeze(1), torch.tensor([0.17, 0.14, 0.11], device=o.device).unsqueeze(0), out)
    return out
    lit = kind > 0
    if lights.shape[0] and lit.any():
        idx = lit.nonzero(as_tuple=True)[0]
        pl, nl = p[idx], nrm[idx]
        dist_all = (lights.unsqueeze(0) - pl.unsqueeze(1)).pow(2).sum(-1).sqrt()
        k = min(n_lights, lights.shape[0])
        near_d, near_i = torch.topk(dist_all, k, dim=1, largest=False)
        acc = torch.zeros_like(pl)
        # Gaussian taper rather than a bounded window. Any window that reaches zero at
        # a finite radius draws that radius on the floor as a visible disc edge, since
        # inverse-square alone still contributes there. exp(-(d/R)^2) never reaches
        # zero, so there is no edge to see, and the k-th light's contribution is
        # negligible before the topk set changes across a surface.
        for j in range(k):
            lp = lights[near_i[:, j]]
            dist = near_d[:, j].clamp(min=1e-3)
            ldir = (lp - pl) / dist.unsqueeze(1)
            ndl = (nl * ldir).sum(1).clamp(min=0.0)
            d_m = dist * CELL_M
            falloff = 1.0 / (1.0 + (d_m * FALLOFF_K) ** 2)
            window = torch.exp(-(d_m / REACH_M) ** 2)
            so = pl + nl * 2e-3 + ldir * 2e-3
            st, skind, _ = dda(so, ldir, occ_t, k_steps=90,
                               t_far=float(dist.max()) + 1.0)
            shadow = ((skind > 0) & (st < dist - 2e-2)).float()
            acc = acc + (ndl * falloff * window * (1.0 - shadow)).unsqueeze(1) * \
                torch.tensor([1.0, 0.58, 0.24], device=o.device).unsqueeze(0) * INTENSITY
        col[idx] = alb[idx] * (acc + FILL)

    # ---- the flames themselves, as emissive blobs along the ray
    if lights.shape[0]:
        flames = lights
        if flames.shape[0] > FLAME_CAP:
            d0 = (flames - o[0].unsqueeze(0)).pow(2).sum(-1)
            flames = flames[torch.topk(d0, FLAME_CAP, largest=False).indices]
        oc = o.unsqueeze(1) - flames.unsqueeze(0)
        b = (oc * d.unsqueeze(1)).sum(-1)
        c_ = (oc * oc).sum(-1) - 0.10 ** 2
        disc = b * b - c_
        tflame = -b - torch.sqrt(disc.clamp(min=0))
        visible = (disc > 0) & (tflame > 0) & (tflame < t.unsqueeze(1))
        glow = (visible.float() * torch.exp(-(tflame.clamp(min=0)) * 0.05)).sum(1).clamp(max=1)
        col = col + glow.unsqueeze(1) * torch.tensor([1.0, 0.78, 0.42], device=o.device).unsqueeze(0)

    col = col / (col + 0.85)
    return col.clamp(0, 1)


# ---------------------------------------------------------------- cameras
def camera_basis(yaw, pitch):
    """Camera-to-world rotation in OpenCV convention: columns are [right, DOWN, forward].

    This is the one convention in play, and it is not a free choice. gsplat takes
    inv(camtoworld) with +Y as image-down, and the official rasteriser hands the model's
    own poses straight to it with no flip, so the reconstruction network was conditioned
    the same way. Building the rig as [right, UP, forward] instead -- the natural thing,
    and what the old renderer did -- is a left-handed camera basis: it has determinant
    -1, renders vertically mirrored, and quietly feeds the model poses that disagree
    with the images they belong to.

    In a right-handed, Y-up world a camera looking along +Z with image-down along world
    -Y necessarily has image-right along world -X. That mirror is the price of the
    convention, and it is harmless: the raycaster, the pose priors and the rasteriser
    all live in it together.
    """
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    f = np.array([sy * cp, sp, cy * cp], dtype=np.float64)          # forward
    r = np.array([-cy, 0.0, sy], dtype=np.float64)                  # image right
    d = np.cross(f, r)                                             # image down
    R = np.stack([r, d, f], axis=1)
    return R.astype(np.float32)


def c2w(yaw, pitch, pos_m):
    """4x4 camera-to-world in OpenCV convention, translation in metres."""
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = camera_basis(yaw, pitch)
    m[:3, 3] = np.asarray(pos_m, dtype=np.float32)
    return m


def _ray_dirs(width, height, fov_x, yaw, pitch, device):
    """Pinhole rays in the same OpenCV frame: +x is image right, +y is image down."""
    fx = (width / 2) / math.tan(fov_x / 2)
    xs = torch.arange(width, device=device).float() - width / 2 + 0.5
    ys = torch.arange(height, device=device).float() - height / 2 + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dirs = torch.stack([xx / fx, yy / fx, torch.ones_like(xx)], -1).reshape(-1, 3)
    R = torch.from_numpy(camera_basis(yaw, pitch)).to(device)
    dirs = dirs @ R.T
    return dirs / dirs.norm(dim=1, keepdim=True), fx


def render_view_tensor(win, pos_world_grid, yaw, pitch,
                       width=504, height=378, fov_x=1.22, chunk=400_000):
    """[3,H,W] float in [0,1] on the GPU -- exactly what the reconstruction wants,
    with no trip through PNG. Position is in WORLD grid units."""
    dirs, _ = _ray_dirs(width, height, fov_x, yaw, pitch, win.occ_t.device)
    lx, ly, lz = win.to_local(pos_world_grid)
    o = torch.tensor([lx, ly, lz], device=win.occ_t.device).float() \
        .unsqueeze(0).expand(dirs.shape[0], 3).contiguous()
    out = [shade(o[i:i + chunk], dirs[i:i + chunk], win)
           for i in range(0, dirs.shape[0], chunk)]
    img = torch.cat(out).reshape(height, width, 3).pow(1 / 2.0).clamp(0, 1)
    return img.permute(2, 0, 1).contiguous()


def render_depth_tensor(win, pos_world_grid, yaw, pitch,
                        width=504, height=378, fov_x=1.22):
    """Ground-truth z-depth in metres, [H,W].

    Colour is the wrong thing to score this world on. A torchlit vault averages ~0.09
    brightness, so a flat dark frame reproduces it to within a few percent and PSNR
    rewards blur: a 19k-splat tile with almost no structure left scored ~39 dB, higher
    than a tile that actually resolved the brickwork. Depth cannot be faked that way --
    it is what the geometry is, regardless of how dim the scene is lit.
    """
    dev = win.occ_t.device
    fx = (width / 2) / math.tan(fov_x / 2)
    xs = torch.arange(width, device=dev).float() - width / 2 + 0.5
    ys = torch.arange(height, device=dev).float() - height / 2 + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cam = torch.stack([xx / fx, yy / fx, torch.ones_like(xx)], -1).reshape(-1, 3)
    ln = cam.norm(dim=1, keepdim=True)
    R = torch.from_numpy(camera_basis(yaw, pitch)).to(dev)
    dirs = (cam / ln) @ R.T
    lx, ly, lz = win.to_local(pos_world_grid)
    o = torch.tensor([lx, ly, lz], device=dev).float().unsqueeze(0).expand(dirs.shape[0], 3).contiguous()
    t, kind, _ = dda(o, dirs, win.occ_t)
    t = torch.where(torch.isinf(t), torch.full_like(t, 0.0), t)
    # t is distance along the ray in grid units; z-depth in metres is the component
    # along the optical axis, which is t/|cam| since the ray was normalised
    z = t / ln.squeeze(1) * CELL_M
    z = torch.where(kind > 0, z, torch.zeros_like(z))
    return z.reshape(height, width)


def render_view(win, pos_world_grid, yaw, pitch, width=512, height=384,
                fov_x=1.22, ss=1):
    """Same view as uint8 RGB, for writing to disk and for eyeballing."""
    img = render_view_tensor(win, pos_world_grid, yaw, pitch,
                             width * ss, height * ss, fov_x)
    if ss > 1:
        img = img.permute(1, 2, 0).reshape(height, ss, width, ss, 3).mean(dim=(1, 3))
    else:
        img = img.permute(1, 2, 0)
    return (img.cpu().numpy() * 255).astype(np.uint8)


def render_pano(win, pos_world_grid, width=1920, height=960, ss=2, chunk=400_000):
    """Equirectangular seed panorama, for the offline HY-World worldgen path."""
    dev = win.occ_t.device
    w, h = width * ss, height * ss
    lon = (torch.arange(w, device=dev).float() + 0.5) / w * 2 * math.pi - math.pi
    lat = math.pi / 2 - (torch.arange(h, device=dev).float() + 0.5) / h * math.pi
    lat_g, lon_g = torch.meshgrid(lat, lon, indexing="ij")
    d = torch.stack([lat_g.cos() * lon_g.sin(), lat_g.sin(), lat_g.cos() * lon_g.cos()],
                    -1).reshape(-1, 3)
    lx, ly, lz = win.to_local(pos_world_grid)
    o = torch.tensor([lx, ly, lz], device=dev).float().unsqueeze(0).expand(d.shape[0], 3).contiguous()
    out = [shade(o[i:i + chunk], d[i:i + chunk], win) for i in range(0, d.shape[0], chunk)]
    img = torch.cat(out).reshape(h, w, 3)
    img = img.reshape(height, ss, width, ss, 3).mean(dim=(1, 3)).pow(1 / 2.0)
    return (img.cpu().numpy() * 255).astype(np.uint8)
