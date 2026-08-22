#!/usr/bin/env python
"""Tiles: the unit of 3DGS the endless world is streamed in.

A tile is a span x span block of cells on a global lattice. Everything here exists to
make two tiles reconstructed minutes apart, by separate forward passes, line up in one
metric world:

  origin      a tile's metric origin is a pure function of its lattice index, so the
              client places it by arithmetic and never by registration
  metric      every forward pass self-normalises to its own frame and scale, and
              conditioning on poses steers that without pinning it. The predicted
              cameras come back with the predictions, so the similarity carrying them
              onto the real rig is solved rather than assumed -- see align_prediction
  footprint   a tile reconstructs well past its own edge, so neighbours would stack
              overlapping copies of the same corridor at slightly different scales.
              Each tile is cropped to the cells it owns: one cell of geometry per cell
              of world

The pruning below mirrors the official worldgen PLY writer rather than inventing its
own: _build_gs_ply_data() keeps splats at or below the 0.98 quantile of max scale, and
thinning is by voxel, never at random.
"""
import math
from dataclasses import dataclass, asdict

import numpy as np
import torch

from infinite_maze import CELL_M, WALL_H, EYE_H, cell_centre_grid, N, E, S, W
import render as R


def snap14(v):
    """The reconstruction ViT uses 14 px patches, so both image dimensions must be
    multiples of 14 -- the official CLI default 952 is 14*68. Anything else asserts
    partway through the forward pass."""
    return max(14, int(v) // 14 * 14)


@dataclass
class TileSpec:
    """Everything the sweep in bench_config.py is allowed to vary.

    The rig is parametrised as positions x azimuths rather than as a single view count,
    because a view count alone hides the thing that actually matters. Distributing N
    views over span^2 cells gave 2 azimuths per cell at N=18 -- and, since the old rig
    tied pitch to sin(2*yaw), those two azimuths were 0 and pi, where sin(2*yaw) is
    zero: the rig never once looked up or down. N=12 and N=18 then produced byte
    identical tiles, which is how the degeneracy showed up at all.
    """
    # span 2, not 3: the rig places n_pos cameras among span^2 cells, so at span 3 with
    # n_pos 4 only four of nine cells were ever observed. Cells with no camera that were
    # also walled off from the ones that had one came back with LITERALLY zero splats --
    # measured 0, 0 and 1,522 against ~63,000 in the observed cells. Those were the holes
    # in the walls. At span 2 every cell has a camera by construction, and the minimum
    # density per cell rises about tenfold.
    span: int = 2                # cells per side in one reconstruction
    n_pos: int = 4               # camera positions inside the tile: span^2, i.e. all
    n_az: int = 4                # azimuths at each position
    # 4 azimuths at 420 px PLUS 4 context cameras -- see n_context below, which is the
    # single largest quality change measured here. Against the previous best (8 azimuths,
    # no context, 364 px) this is better on every axis AND faster, on FEWER views (28 vs
    # 32): accuracy 23.1 -> 14.6 cm, splats within 5 cm of a real surface 19.7 -> 33.6%,
    # true surface covered 43.1 -> 61.2%, and 1.38 -> 1.50 m/s. A camera outside the tile
    # looking in is worth more than another azimuth inside it.
    #
    # Kept for the record, because it was wrong twice over:
    #
    # Under the camera-derived scale, more views measured WORSE (6 -> 8 azimuths took
    # depth from 11.2 to 20.8 cm) and I concluded the model was being pushed past its
    # trained regime. That was the broken scale talking: a noisier scale estimate hurt
    # more than the extra views helped. With the scale measured against the raycaster,
    # 32 views is best on every axis of the surface audit, and accuracy turns out to be
    # roughly FLAT in view count while COVERAGE scales strongly with it -- 4x2=8 views
    # covers 24.7% of the true surface, 4x8=32 covers 52.8%.
    #
    # Spatial diversity matters far more than azimuths: 1 position x 8 azimuths is a
    # disaster (41.1 cm, 4.6% covered) because it has almost no parallax, while 4x2 with
    # the same 8 views gets 20.7 cm and 24.7%.
    #
    # 364 px rather than 504 buys the speed those extra views cost. Against the previous
    # 6x504 default this is better on ALL FOUR quality measures and faster:
    # acc 26.2 -> 23.2 cm, inlier 19.2 -> 19.8%, cov 8.8 -> 6.8 cm,
    # covered 39.1 -> 42.9%, and 1.32 -> 1.58 m/s. 4x8 at 504 is better still
    # (17.9 cm, 52.8% covered) but runs at 1.00 m/s, under walking pace.
    #
    # Pushing for complete walls (102 deg fov, 8 azimuths, 364 px) did cut holes to 3.1%,
    # and made the walls worse: depth error went to 43.4 cm with only 5.2% of pixels
    # within 10 cm, so the splats no longer sat on one plane and a wall read as a cloud of
    # fragments -- "scattered". Coverage is not the only thing a wall needs; it needs to be
    # a surface. This setting is 19.4 cm / 22.3% within 10 cm at 9.2% empty and 1.45 m/s,
    # which keeps ahead of a walk. 1.22 fov is better still on depth (13.5 cm) but leaves
    # 14.2% holes; 8 azimuths at 504 is better on both (15.6 cm, 6.2%) and manages only
    # 1.03 m/s, under walking pace.
    size: int = 476              # variant G: official 480 snapped to ViT-14
    # 1.50 rad (70 deg vertical), not 1.22 (55 deg). The rig's vertical field was the
    # reason the floor and ceiling came back empty: each view saw only a narrow band of
    # them, and 30% of the lower frame / 25% of the upper had no geometry at all. Tilting
    # the rig further made it worse, and texturing the ceiling did nothing -- the surfaces
    # were being seen, just barely. Widening the field fixes it at no extra pixels:
    # empty 14.7% -> 10.0%, floor holes 30% -> 18%, and depth error IMPROVES 14.4 -> 12.1
    # cm. That last part is what separates this from relaxing scale_quantile, which also
    # filled holes but by smearing, with depth unchanged. The cost is angular resolution:
    # edge energy against ground truth falls from 0.99x to 0.82x.
    fov_x: float = 2.0944        # variant G: official 120 deg
    # 400k voxel-prunes to ~340k, which is 18.2 MB on the wire and fits the browser
    # renderer's slot. Pruning is not a bandwidth compromise: unpruned tiles measure
    # 113.8 cm median depth error against 16.6 cm here, because the splat head emits one
    # Gaussian per pixel per view and the duplicates z-fight rather than reinforce.
    max_splats: int = 400_000    # 0 = no budget
    # 0.98, the official writer's value, kept after examining the alternative.
    #
    # The filter drops Gaussians stretched across depth discontinuities and is doing real
    # work: disabling it fills every hole and takes depth error from 14.4 cm to 200.6 cm.
    # Relaxing it to 0.995 is subtler and looked attractive -- 14.7% -> 10.6% of the frame
    # with no geometry, at +0.2 cm depth. But the splats it lets back in fill those holes
    # BY SMEARING: edge energy falls from 0.99x of ground truth to 0.71x. That is the same
    # trade that made colour PSNR pick a 19k-splat blur over a correct tile, and it is
    # refused for the same reason -- blur degrades every pixel, a hole degrades one.
    scale_quantile: float = 0.98
    # Swept, and 1.0 won: 20.28 dB against 19.89 / 19.41 / 18.97 / 18.41 at gains of
    # 0.8 / 0.6 / 0.45 / 0.3. The model's own opacities are the right ones.
    #
    # Worth recording why this was ever in doubt. Running opacities through a second
    # sigmoid scored ~7 dB better in calibration, which looked like evidence that the
    # field wanted softening -- the head emits one Gaussian per pixel per view, so at
    # n_pos*n_az views every surface is represented that many times over. But a second
    # sigmoid is not a softening: on values already in [0.14, 0.99] it maps to
    # [0.54, 0.73], RAISING the transparent splats as much as it lowers the opaque ones.
    # A gain only lowers, and lowering monotonically hurts. So the 7 dB came from
    # compressing the range, not from reducing opacity, and it was measured on a rig view
    # where the redundancy is aligned with the camera. Neither transform is applied.
    opacity_gain: float = 1.0
    # Where a splat's POSITION comes from, and the single largest geometry win here.
    #
    # "pred" is what the model does at inference: unproject gs_depth with its OWN
    # predicted cameras. Those are not good. Over 391 tiles the fit between predicted and
    # true camera centres had a median residual of 12.6 cm, and on a rig that is level by
    # construction -- every camera at exactly EYE_H -- the model invents 31 cm of vertical
    # camera spread. All of that error lands in every unprojected point, which is what
    # makes a wall arrive as a cloud rather than a surface.
    #
    # "gt" unprojects with the rig poses we already know exactly, taking only the global
    # SCALE from the camera fit: one number instead of 24 noisy poses. This is what the
    # model's own TRAINING path does (position_from="gsdepth+gtcamera"); only inference
    # substitutes its estimates. Measured over three separate places in the world, depth
    # error falls 31.7 -> 23.8 cm, a 24.9% improvement (+35.3% / +25.8% / +18.5%), and
    # pixels within 10 cm rise from 17.0% to 20.2%. Costs nothing: 1.85s vs 1.93s a tile.
    unproject: str = "gt"        # "pred" | "gt"
    # Drop the least confident depths before anything else. The model emits
    # gs_depth_conf per pixel and the pipeline has apply_confidence_filter for exactly
    # this -- but at inference it is unreachable: prepare_splats returns at
    # `if is_inference` two lines before the filter is consulted, and enable_conf_filter
    # defaults to False anyway. Low-confidence depth is where a splat lands off the
    # surface, so this is aimed straight at scatter. 0 = keep everything.
    conf_percent: float = 0.0    # discard this % of the least confident splats
    # How the model's depth is converted to metres. "camera" takes the scale from the
    # similarity fit against the predicted cameras -- which is what it was doing, and the
    # scale then wandered 3.85 to 5.40 between tiles, a 40% spread, so neighbouring tiles
    # were built at different sizes. "truth" measures it against the raycaster's own depth
    # for views we drew ourselves, so it costs one cheap depth raycast per probe view.
    # "surface" refines the depth-ratio scale by directly minimising how far the
    # resulting points land from the walls we know are there. Fitting depth ratios is a
    # proxy; this optimises the thing actually wanted. It matters because the residual is
    # not uniform across tiles: F-score correlates -0.65 with the solved scale and -0.51
    # with floor error, and on the bad tiles the floor still sits 10-12 cm out AFTER the
    # ratio fit, i.e. one global ratio does not fit those tiles.
    #
    # Cheap because p(s) = C + s*v: the camera centre C and the ray offset v are fixed, so
    # a candidate scale is one lookup per point, not a re-unprojection.
    # "perview" fits a separate scale to EACH view against its own raycast depth.
    # Motivated by the diagnostic: F-score correlates -0.65 with the solved scale and
    # -0.51 with floor error, and on bad tiles the floor is still 10-12 cm out after a
    # global fit -- i.e. no single number fits them. Per-view stays coherent because each
    # view's points are unprojected along its own rays from its own known pose.
    #
    # "surface" (fit the scale by minimising distance to the known walls) was tried and is
    # WORSE -- F 28.3% vs 31.1%, floor error 9.4 vs 5.6 cm -- because that objective is
    # minimised by degenerate scales that compress geometry onto the nearest surface.
    scale_from: str = "perview"  # "perview" | "truth" | "surface" | "camera"
    # Measured over 12 tiles across the world, per-view against one global scale:
    # F@5cm 31.1 -> 40.2%, WORST TILE 17.5 -> 32.9%, accuracy 25.9 -> 19.8 cm,
    # coverage 40.9 -> 52.5%, floor error 5.6 -> 3.2 cm, at 1.50 vs 1.53 m/s.
    # The worst-tile number is the point: a global scale left some tiles badly stretched,
    # and a stretched tile is what reads on screen as scattered.
    depth_probe_every: int = 4   # raycast true depth for one view in this many
                                 # (perview forces this to 1)
    # Cameras placed OUTSIDE the tile, looking back into it. Every chunked 3DGS system
    # in the literature gives a chunk more input than its own footprint -- VastGaussian
    # expands the cell 20% and adds visibility-selected distant cameras, Hierarchical
    # 3DGS takes cameras out to 2x the chunk bounds, Block-NeRF overlaps blocks 50% --
    # and then crops the OUTPUT back. Removing that support is what VastGaussian's
    # ablation shows collapsing SSIM 0.882 -> 0.694. This rig had no such support: every
    # camera sat inside the tile it was reconstructing, which starves exactly the
    # boundary where neighbouring tiles have to agree.
    n_context: int = 4           # camera positions in the ring of cells around the tile
    n_az_context: int = 3        # azimuths at each, swept about the bearing to the tile
    # How a surviving Gaussian inherits the opacity of the ones it replaced.
    #   "voxel"  alpha_v = 1 - prod_i(1 - a_i) over the splats actually in that voxel
    #   "global" 1 - (1-a)^k with one k = n_before/n_kept for the whole tile
    #   "none"   leave it, which is what left walls see-through
    # "global" is RoSplat Eq.5 (arXiv 2605.13093) with m/m~ = k -- the same algebra, in the
    # same feed-forward regime, derived from the same W = 1 - prod(1-a). Long-LRM describes
    # this failure exactly: "the model tends to diffuse an opaque color into multiple
    # translucent Gaussians", so pruning transparent Gaussians leaves objects incomplete.
    #
    # "voxel" is the more faithful operation and MEASURED WORSE: 16.6% see-through against
    # global's 14.3%, accumulated alpha 0.867 vs 0.878. It reproduces exactly what the
    # pre-prune field composited -- but that field only reached alpha 0.891 itself, so
    # faithfully reproducing it preserves the transparency. Global k over-corrects, and the
    # over-correction is the point. Kept selectable because "correct" losing to
    # "deliberately too strong" is the kind of result that should stay reproducible.
    # Reject splats further than this from ANY real surface. In a reconstruction of an
    # unknown scene this would be impossible; here the labyrinth is ours, so a splat
    # floating in the middle of a corridor is provably wrong rather than merely unlikely.
    # It is aimed at the scatter the crop concentrates: raw unprojected points sit 9.0 cm
    # from a real surface, and after cropping to the tile's own volume that doubles to
    # 19.2 cm, because the crop keeps the floaters inside the footprint and discards the
    # accurate points resting on neighbouring surfaces. 0 disables it.
    # 10 cm, and deliberately not tighter. The rejection threshold and the audit's
    # tolerance are the SAME operation, so driving it toward 5 cm would push measured
    # precision to 100% by construction -- the filter would be grading itself. 10 cm keeps
    # a 2x margin, and the justification rests on COVERAGE (60.8 -> 62.0%), which deleting
    # splats can only hurt: the gain is the fixed 400k budget being spent on real surface
    # instead of on ~160k floaters.
    surface_reject_cm: float = 10.0
    # Which splat in a voxel survives, and where it is put.
    #   "opacity" the most opaque one -- what this did, and opacity says nothing about
    #             whether the splat is in the right PLACE
    #   "conf"    the one the model was most confident about geometrically
    #   "mean"    keep the most confident one's appearance but move it to the MEAN
    #             position of the voxel's splats: averaging k noisy positions cuts the
    #             position error by about sqrt(k), which picking one cannot do
    voxel_pick: str = "opacity"
    opacity_merge: str = "global"
    # sqrt(k) growth of the survivor's radius. Undocumented in the 3DGS pruning literature
    # and RoSplat reaches brightness invariance without it -- but removing it here costs
    # a lot: see-through goes 16.6% -> 26.1%. It stays, on measurement rather than on
    # precedent.
    scale_grow: bool = True
    view_margin: int = 8         # extra cells of window, so corridors are not cut short
    pitch_amp: float = 0.45      # radians the rig tilts up and down
    baseline: float = 0.35       # metres the rig steps off the cell centre

    @property
    def views(self):
        return self.n_pos * self.n_az

    # Aspect of the views fed to the model. The official path feeds SQUARE images --
    # prepare_images_to_tensor resizes and centre-crops to target_size in BOTH dims
    # (518 by default, 952 in the CLI) -- so the network has most likely only ever seen
    # square input. This rig has been feeding 4:3 the whole time.
    aspect: float = 1.0          # variant G: official square

    def wh(self):
        return snap14(self.size), snap14(self.size * self.aspect)

    def key(self):
        return (f"s{self.span}_{self.n_pos}x{self.n_az}_p{self.size}"
                f"_m{self.max_splats}")


# ---------------------------------------------------------------- lattice
def tile_cells(tc, tr, span):
    return tc * span, tr * span


def tile_origin_m(tc, tr, span):
    """Metric centre of the tile, in world metres. Pure arithmetic on the index --
    this is what lets the client drop a tile into place with no registration."""
    c0, r0 = tile_cells(tc, tr, span)
    gx = np.mean([cell_centre_grid(c0 + i, 0)[0] for i in range(span)])
    gz = np.mean([cell_centre_grid(0, r0 + j)[1] for j in range(span)])
    return float(gx * CELL_M), float(gz * CELL_M)


def tile_of_cell(c, r, span):
    return c // span, r // span


def tile_of_pos_m(x_m, z_m, span):
    """Which tile a player standing at (x,z) metres is inside.

    The subtrahend is 1, not 1.5. Open cell (c,r) occupies grid index (2c+1, 2r+1), so
    it spans gx in [2c+1, 2c+2) and the inverse of that interval is floor((gx-1)/2).
    Subtracting the CENTRE offset 1.5 instead partitions on cell centres, which puts the
    western quarter of every cell -- 25% of all positions -- in cell c-1. The client's
    cellAt had the identical slip, so the two agreed with each other while both disagreed
    with the geometry, and nothing looked wrong. It is not cosmetic here: this is what
    centres /ring, so for a quarter of every corridor the ring was centred one tile
    behind the player, spending its radius on tiles already walked through instead of
    the one being walked into."""
    gx, gz = x_m / CELL_M, z_m / CELL_M
    return int((gx - 1) // 2) // span, int((gz - 1) // 2) // span


# ---------------------------------------------------------------- camera rig
def render_rig(maze, tc, tr, spec, device=R.DEV):
    """Views for one tile, as a GPU tensor, plus the poses they came from.

    The PNG round trip (encode -> /dev/shm -> decode -> resize) was half the view cost
    and the model decodes straight back to a tensor anyway. Rendering at exactly the
    model's target size means no resize either, so the raycast output reaches the
    network without ever leaving the GPU.

    Poses are expressed relative to the TILE ORIGIN, so the reconstruction comes back
    centred on the tile and the client's placement is a translation by tile_origin_m.
    """
    W, H = spec.wh()
    span = spec.span
    c0, r0 = tile_cells(tc, tr, span)
    fx = (W / 2) / math.tan(spec.fov_x / 2)

    win = R.build_window(maze, c0 - spec.view_margin, r0 - spec.view_margin,
                         span + 2 * spec.view_margin, span + 2 * spec.view_margin,
                         device=device)

    cells = [(c0 + i, r0 + j) for j in range(span) for i in range(span)]
    gx_c = np.mean([cell_centre_grid(c, 0)[0] for c, _ in cells])
    gz_c = np.mean([cell_centre_grid(0, r)[1] for _, r in cells])

    # positions spread over the tile's cells; azimuths sweep the full circle at each
    picks = (list(range(len(cells))) if spec.n_pos >= len(cells) else
             sorted(set(np.linspace(0, len(cells) - 1, spec.n_pos).round().astype(int))))
    rig = [(cells[ci], None) for ci in picks]
    if spec.n_context > 0:
        # one ring of cells around the tile, looking back in
        ring = [(c, r)
                for r in range(r0 - 1, r0 + span + 1)
                for c in range(c0 - 1, c0 + span + 1)
                if not (c0 <= c < c0 + span and r0 <= r < r0 + span)]
        if ring:
            # Pick these by BEARING from the tile centre, not by position in the list.
            #
            # `ring` is built row-major, so spreading indices evenly over it is uniform
            # in list order and badly non-uniform in direction. For span=2, n_context=4
            # it chose (-1,-1), (-1,0), (2,1), (2,2): the west and east mid-side cells
            # plus two diagonal corners, and never once a cell directly north or south
            # of the tile -- the four that would look straight down the +z/-z seams,
            # (0,-1), (1,-1), (0,2), (1,2), were unreachable by that rule.
            #
            # A tile's boundary strip is reconstructed almost entirely from the context
            # camera facing it, so the x seams had one and the z seams did not, and it
            # shows: measured against the union of the resident ring, x seams land 85-99%
            # of their surface within 5 cm while z seams manage 52-62%, with 12-14% of it
            # further than 25 cm from any splat. Same camera count, same cost -- the
            # views were simply all pointing along one axis.
            # ...and among the cells near a bearing, prefer one that can actually SEE
            # into the tile. A context camera in a cell walled off from the tile spends
            # its views on the far side of that wall and contributes nothing to the seam
            # it was placed to cover.
            def facing(cell):
                c, r = cell
                d = []
                if c < c0: d.append(E)
                if c >= c0 + span: d.append(W)
                if r < r0: d.append(S)
                if r >= r0 + span: d.append(N)
                return d

            targets = [-math.pi + 2 * math.pi * t / spec.n_context
                       for t in range(spec.n_context)]
            left = list(ring)
            for want in targets:
                if not left:
                    break
                def rank(cell, want=want):
                    gx, gz = cell_centre_grid(cell[0], cell[1])
                    a = math.atan2(gz - gz_c, gx - gx_c)
                    gap = abs((a - want + math.pi) % (2 * math.pi) - math.pi)
                    blind = all(maze.has_wall(cell[0], cell[1], d) for d in facing(cell))
                    return (1 if blind else 0, gap)
                best = min(left, key=rank)
                left.remove(best)
                rig.append((best, "context"))
    imgs, c2ws, depths = [], [], []
    for k, (cell, kind) in enumerate(rig):
        cc, rr = cell
        ox, oz = cell_centre_grid(cc, rr)
        n_az = spec.n_az if kind is None else spec.n_az_context
        # a context camera outside the tile only has to see INTO it, so its azimuths are
        # swept about the bearing to the tile centre rather than round the full circle
        aim = (np.arctan2(gx_c - ox, gz_c - oz) if kind == "context" else 0.0)
        for j in range(n_az):
            # stagger the azimuth phase per position so the tile is not sampled from
            # the same set of directions everywhere, and cycle pitch through
            # down/level/up so floor and ceiling are actually observed
            yaw = (aim + (j - (n_az - 1) / 2) * (2.0 / max(n_az, 1))
                   if kind == "context"
                   else 2 * np.pi * (j / n_az) + 0.37 * k)
            pitch = 0.0  # variant G: full level ring; down coverage comes from the nadir + corner cams below
            cy, sy = np.cos(yaw), np.sin(yaw)
            bgl = spec.baseline / CELL_M
            px, pz = ox + sy * bgl, oz + cy * bgl
            imgs.append(R.render_view_tensor(win, (px, EYE_H, pz), float(yaw),
                                             float(pitch), width=W, height=H,
                                             fov_x=spec.fov_x))
            # We drew these views, so their true depth is free. Keep a few: it is what
            # pins the model's depth scale to metres without trusting its cameras.
            _every = 1 if getattr(spec, 'scale_from', '') == 'perview' else max(1, spec.depth_probe_every)
            if len(imgs) % _every == 1 or _every == 1:
                depths.append((len(imgs) - 1,
                               R.render_depth_tensor(win, (px, EYE_H, pz), float(yaw),
                                                     float(pitch), width=W, height=H,
                                                     fov_x=spec.fov_x)))
            c2ws.append(R.c2w(float(yaw), float(pitch),
                              ((px - gx_c) * CELL_M, EYE_H * CELL_M,
                               (pz - gz_c) * CELL_M)))

            if kind is None:
                # one nadir per interior position (variant G): the floor is the
                # player's ground truth and the level ring never sees it straight on
                imgs.append(R.render_view_tensor(win, (ox, EYE_H, oz), float(aim),
                                                 -spec.pitch_amp, width=W, height=H,
                                                 fov_x=spec.fov_x))
                depths.append((len(imgs) - 1,
                               R.render_depth_tensor(win, (ox, EYE_H, oz), float(aim),
                                                     -spec.pitch_amp, width=W, height=H,
                                                     fov_x=spec.fov_x)))
                c2ws.append(R.c2w(float(aim), -spec.pitch_amp,
                                  ((ox - gx_c) * CELL_M, EYE_H * CELL_M,
                                   (oz - gz_c) * CELL_M)))

                # variant J: corner cameras. Two walls meeting = blind pocket for the
                # centre-standing rig. One view per recessed corner, aimed in, pitched down.
                cc, rr = cell
                for dc, dr, dN, dE in ((0.5, -0.5, 0, 1), (0.5, 0.5, 2, 1),
                                       (-0.5, 0.5, 2, 3), (-0.5, -0.5, 0, 3)):
                    if maze.has_wall(cc, rr, dN) and maze.has_wall(cc, rr, dE):
                        gx, gz = ox + dc, oz + dr
                        _yawc = float(np.arctan2(gx - ox, gz - oz))
                        imgs.append(R.render_view_tensor(win, (ox, EYE_H, oz), _yawc,
                                                         -0.5, width=W, height=H,
                                                         fov_x=spec.fov_x))
                        depths.append((len(imgs) - 1,
                                       R.render_depth_tensor(win, (ox, EYE_H, oz), _yawc,
                                                             -0.5, width=W, height=H,
                                                             fov_x=spec.fov_x)))
                        c2ws.append(R.c2w(_yawc, -0.5,
                                          ((ox - gx_c) * CELL_M, EYE_H * CELL_M,
                                           (oz - gz_c) * CELL_M)))

    batch = torch.stack(imgs).unsqueeze(0)                       # [1,N,3,H,W]
    extr = torch.from_numpy(np.stack(c2ws)).unsqueeze(0)         # [1,N,4,4]
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=np.float32)
    intr = torch.from_numpy(np.stack([K] * len(imgs))).unsqueeze(0)
    # c2ws[0] is returned for callers that want the anchor pose; placement itself uses
    # the whole rig, since align_prediction fits against every camera rather than one.
    return batch, extr, intr, win, c2ws[0], depths


def eval_cameras(maze, tc, tr, spec, n=6, device=R.DEV, at_seam=True):
    """Held-out cameras for scoring a tile, with ground truth rendered by the raycaster.

    Placed on the tile's edge by default: a seam is where two independently normalised
    reconstructions have to agree, so it is the only place worth measuring. Scoring the
    middle of a tile flatters every configuration equally.
    """
    W, H = spec.wh()
    span = spec.span
    c0, r0 = tile_cells(tc, tr, span)
    win = R.build_window(maze, c0 - spec.view_margin, r0 - spec.view_margin,
                         span + 2 * spec.view_margin, span + 2 * spec.view_margin,
                         device=device)
    gx_c = np.mean([cell_centre_grid(c0 + i, 0)[0] for i in range(span)])
    gz_c = np.mean([cell_centre_grid(0, r0 + j)[1] for j in range(span)])

    edge = [(c0, r0 + span - 1), (c0 + span - 1, r0), (c0, r0),
            (c0 + span - 1, r0 + span - 1)] if at_seam else \
           [(c0 + span // 2, r0 + span // 2)]
    out = []
    for i in range(n):
        cc, rr = edge[i % len(edge)]
        ox, oz = cell_centre_grid(cc, rr)
        # deliberately off the rig's own yaws and off its baseline circle
        yaw = 2 * np.pi * (i / n) + 0.31
        pitch = 0.18 * np.sin(3.0 * yaw)
        gt = R.render_view_tensor(win, (ox, EYE_H, oz), float(yaw), float(pitch),
                                  width=W, height=H, fov_x=spec.fov_x)
        m4 = R.c2w(float(yaw), float(pitch),
                   ((ox - gx_c) * CELL_M, EYE_H * CELL_M, (oz - gz_c) * CELL_M))
        out.append({"gt": gt, "c2w": m4, "yaw": float(yaw), "pitch": float(pitch),
                    "cell": (cc, rr)})
    return out, win


# ---------------------------------------------------------------- splat post-process
def quantile(t, q, cap=8_000_000):
    """torch.quantile with the size limit worked around.

    torch.quantile refuses tensors past roughly 16M elements, and a tile at 952 px with
    54 views carries 36.7M splats -- so the crash arrives exactly at the settings worth
    testing. A uniform subsample estimates a quantile to well under the precision any of
    these thresholds need, and the subsample is deterministic so tiles stay reproducible.
    """
    if t.numel() <= cap:
        return torch.quantile(t, q)
    step = t.numel() // cap + 1
    return torch.quantile(t.flatten()[::step].contiguous(), q)


def gaussians_from_predictions(pred):
    if "splats" not in pred:
        raise RuntimeError("model returned no splats head")
    sp = pred["splats"]
    return (sp["means"][0].reshape(-1, 3).float(),
            sp["scales"][0].reshape(-1, 3).float(),
            sp["quats"][0].reshape(-1, 4).float(),
            (sp.get("sh", sp.get("colors"))[0]).reshape(-1, 3).float(),
            sp["opacities"][0].reshape(-1, 1).float())


def umeyama(src, dst):
    """Least-squares similarity transform (scale, rotation, translation) src -> dst.

    Closed form, with the reflection guard: without the determinant correction a noisy
    fit can return an improper rotation, which mirrors the tile instead of turning it.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    C = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - s * (R @ mu_s)
    rmse = float(np.sqrt((((s * (R @ src.T).T + t) - dst) ** 2).sum(1).mean()))
    return s, R.astype(np.float32), t.astype(np.float32), rmse


def scale_from_true_depth(pred, depths, lo=0.05):
    """Pin the model's depth to metres against depth we already know exactly.

    The alternative is the scale from the camera similarity fit, and that scale moved
    3.851 -> 4.871 between two tiles of the same world -- a 26% disagreement with the
    truth, which puts one tile's floor 35 cm out while its neighbour's is fine. Pooling
    the ratio over all valid pixels of a few probe views gives one robust number per tile.
    """
    gsd = pred["gs_depth"][0].float()
    ratios = []
    for idx, td in depths:
        if idx >= gsd.shape[0]:
            continue
        md = gsd[idx].reshape(td.shape)
        m = (td > lo) & (md > 1e-4)
        if m.any():
            ratios.append((td[m] / md[m]).flatten())
    if not ratios:
        return None
    return float(torch.cat(ratios).median())


def refine_scale_on_surfaces(pred, extr, intr, maze, tc, tr, span, s0,
                             lo=0.75, hi=1.30, iters=18, sample=40000):
    """Golden-section search for the scale that puts the points on the real walls."""
    import sys as _s
    _s.path.insert(0, "/home/zrx/HY-World-2.0")
    from hyworld2.worldrecon.hyworldmirror.models.utils.geometry import (
        depth_to_world_coords_points)
    d = pred["gs_depth"][0].float()
    S, H, W = d.shape[0], d.shape[1], d.shape[2]
    dep = d.reshape(S, H, W)
    e = extr[0].to(dep.device).float()
    k = pixel_centre_K(intr[0].to(dep.device).float())
    p1, _, _ = depth_to_world_coords_points(dep * 1.0, e, k)
    p1 = p1.reshape(-1, 3)
    C = e[:, :3, 3].unsqueeze(1).expand(S, H * W, 3).reshape(-1, 3)
    v = p1 - C                                    # p(s) = C + s*v
    idx = torch.randperm(v.shape[0], device=v.device)[:sample]
    C, v = C[idx], v[idx]

    def cost(s):
        pts = C + s * v
        dd = surface_distance(maze, pts, tc, tr, span)
        return float(dd.clamp(max=0.5).median())

    a, b = s0 * lo, s0 * hi
    gr = (5 ** 0.5 - 1) / 2
    c, dd_ = b - gr * (b - a), a + gr * (b - a)
    fc, fd = cost(c), cost(dd_)
    for _ in range(iters):
        if fc < fd:
            b, dd_, fd = dd_, c, fc
            c = b - gr * (b - a); fc = cost(c)
        else:
            a, c, fc = c, dd_, fd
            dd_ = a + gr * (b - a); fd = cost(dd_)
    return float((a + b) / 2)


def scale_per_view(pred, depths, n_views, lo=0.05):
    """One scale per view, from that view's own raycast depth.

    Views without a probe fall back to the median of those that have one."""
    gsd = pred["gs_depth"][0].float()
    per = {}
    for idx, td in depths:
        if idx >= gsd.shape[0]:
            continue
        md = gsd[idx].reshape(td.shape)
        m = (td > lo) & (md > 1e-4)
        if m.any():
            per[idx] = float((td[m] / md[m]).median())
    if not per:
        return None
    fallback = float(np.median(list(per.values())))
    return torch.tensor([per.get(i, fallback) for i in range(n_views)],
                        dtype=torch.float32)


def pixel_centre_K(intr):
    """The same intrinsics, shifted for a helper that samples pixels at integer coords.

    render.py builds ray i as `arange(W) - W/2 + 0.5`, so pixel i is sampled at image
    coordinate i + 0.5 -- the OpenCV convention, for which cx = W/2 is exactly the
    centre. That is the K the model is given, and it is right for the model, because
    the images it is given were drawn with those rays.

    depth_to_camera_coords, which is what turns the model's depth back into points,
    builds its grid as `torch.arange(W)` -- pixel i at coordinate i, no half. Handing
    it cx = W/2 therefore aims every ray half a pixel off, and since it is the same
    half-pixel in both axes, every splat in the tile is displaced by z*sqrt(2)*0.5/fx
    along a fixed diagonal in camera frame: a median 8.6 mm at the served splats' median
    depth, and once the rig's yaws average out, a coherent ~6 mm vertical lift of the
    whole tile. Both conventions are internally fine; sharing one K between them is not.
    """
    k = intr.clone()
    k[..., 0, 2] -= 0.5
    k[..., 1, 2] -= 0.5
    return k


def means_from_true_cameras(pred, extr, intr, scale):
    """Unproject gs_depth with the rig poses we know, instead of the ones the model guessed.

    The model's depth is in its own normalised units, so it still needs the global scale
    from the camera fit -- but only that one number. Per-camera position and rotation come
    from the rig, which is exact, so the pose error cannot enter the points. Returns means
    already in tile frame, which also makes the rigid part of align_prediction unnecessary.
    """
    import sys
    sys.path.insert(0, "/home/zrx/HY-World-2.0")
    from hyworld2.worldrecon.hyworldmirror.models.utils.geometry import (
        depth_to_world_coords_points)
    d = pred["gs_depth"][0].float()                      # [S,H,W,1]
    S, H, W = d.shape[0], d.shape[1], d.shape[2]
    if torch.is_tensor(scale) and scale.numel() > 1:     # one scale per view
        sc = scale.to(d.device).reshape(-1, 1, 1)[:S]
    else:
        sc = float(scale)
    depth = (d.reshape(S, H, W) * sc)                    # model units -> metres
    e = extr[0].to(depth.device).float()                 # [S,4,4] tile frame, exact
    k = pixel_centre_K(intr[0].to(depth.device).float())  # [S,3,3], integer-pixel grid
    pts, _, _ = depth_to_world_coords_points(depth, e, k)
    return pts.reshape(-1, 3)


def align_prediction(g, pred, extr):
    """Put a tile in the world by solving for the frame the model actually used.

    The splat means are unprojected from gs_depth using the model's own PREDICTED
    cameras ("gsdepth+predcamera"), which live in a frame the network normalises for
    itself -- conditioning on poses steers it, it does not pin it. Measured on one tile,
    the predicted rig spanned 1.29 x 0.05 x 1.28 where the real rig spans 8.65 x 0 x
    8.62: same shape, ~6.7x smaller. Assuming the output was already in camera 0's
    frame is what put tiles in the wrong place and at the wrong size, and no amount of
    tuning downstream could have fixed it.

    Since the predicted cameras come back with the predictions, the transform is not a
    guess: fit the similarity that carries predicted camera centres onto the true ones
    and apply it to the Gaussians. The fit residual is returned, so a tile that did not
    align can be detected rather than served.
    """
    means, scales, quats, colors, opac = g
    src = pred["camera_poses"][0].float().detach().cpu().numpy()[:, :3, 3]
    dst = extr[0].detach().cpu().numpy()[:, :3, 3]
    s, R, t, rmse = umeyama(src, dst)
    Rt = torch.as_tensor(R, dtype=torch.float32, device=means.device)
    tt = torch.as_tensor(t, dtype=torch.float32, device=means.device)
    means = (means * s) @ Rt.T + tt
    quats = quat_mul(mat_to_quat(R).to(means.device), quats)
    return [means, scales * s, quats, colors, opac], s, rmse




def mat_to_quat(rot):
    """3x3 rotation -> wxyz quaternion, branching on the largest diagonal term so it
    stays conditioned for the half-turns this camera basis produces."""
    m = np.asarray(rot, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (m[2, 1] - m[1, 2]) / S,
             (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / S, 0.25 * S,
             (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S]
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S,
             0.25 * S, (m[1, 2] + m[2, 1]) / S]
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S,
             (m[1, 2] + m[2, 1]) / S, 0.25 * S]
    return torch.tensor(q, dtype=torch.float32)


def quat_mul(a, b):
    """Hamilton product a (x) b, both wxyz; `a` is one quaternion, `b` is [N,4]."""
    aw, ax, ay, az = (a[i] for i in range(4))
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], dim=1)


def scale_quantile_filter(g, q=0.98, ret_mask=False):
    """Drop the largest splats, exactly as the official PLY writer does.

    Feed-forward reconstruction emits a tail of huge Gaussians stretched across depth
    discontinuities; they smear over everything behind them, which is what makes
    surfaces look low quality. Calling the forward pass directly for speed skips the
    official writer, and skips this with it."""
    if g[0].shape[0] == 0:
        return g, 0
    m = g[1].max(dim=-1)[0]
    keep = m <= quantile(m, q)
    out = [t[keep] for t in g]
    if ret_mask:
        return out, int((~keep).sum()), keep
    return out, int((~keep).sum())


def voxel_prune(g, target, merge="voxel", grow=True, pick="opacity", conf=None):
    """Thin by voxel, as the official pipeline does, not at random.

    Random subsampling removes a surface in clumps and leaves holes. Voxel pruning keeps
    the most opaque splat per voxel, so coverage stays uniform at any budget."""
    means, opac = g[0], g[4].reshape(-1)
    n = means.shape[0]
    if n <= target or n == 0:
        return g, 0.0
    lo = means.min(0).values
    ext = (means.max(0).values - lo).clamp(min=1e-6)
    v = float((ext.prod() / max(target, 1)) ** (1.0 / 3.0))
    key = None
    for _ in range(10):
        idx = ((means - lo) / v).floor().to(torch.int64)
        key = idx[:, 0] * 73856093 ^ idx[:, 1] * 19349663 ^ idx[:, 2] * 83492791
        got = torch.unique(key).numel()
        if 0.75 * target <= got <= 1.1 * target:
            break
        v *= (got / max(target, 1)) ** (1.0 / 3.0)
    order = torch.argsort(opac, descending=True)
    _, inv = torch.unique(key[order], return_inverse=True)
    first = torch.full((int(inv.max()) + 1,), n, dtype=torch.long, device=means.device)
    first.scatter_reduce_(0, inv, torch.arange(inv.numel(), device=means.device),
                          reduce="amin", include_self=True)
    keep = torch.zeros(n, dtype=torch.bool, device=means.device)
    sel = first[first < n]
    keep[order[sel]] = True
    out = [t[keep] for t in g]
    if pick == "mean":
        # average the voxel's member positions rather than adopting one of them
        nvox = int(inv.max()) + 1
        m_ord = means[order]
        acc = torch.zeros(nvox, 3, device=means.device, dtype=torch.float32)
        acc.index_add_(0, inv, m_ord)
        cnt = torch.zeros(nvox, device=means.device, dtype=torch.float32)
        cnt.index_add_(0, inv, torch.ones_like(inv, dtype=torch.float32))
        out[0] = (acc[inv[sel]] / cnt[inv[sel]].unsqueeze(1))
    # Survivors must cover the area of everyone removed or the voxel grid shows through
    # as a lattice of gaps: keep 1 in k -> grow the radius by sqrt(k), and never let a
    # splat be smaller than the voxel it now stands for.
    kept = int(keep.sum())
    if kept:
        if grow:
            out[1] = torch.clamp(out[1] * float(n / kept) ** 0.5, min=v * 0.5)
        else:
            out[1] = torch.clamp(out[1], min=v * 0.5)
        # ...and the same for OPACITY, which this was missing. The head predicts one
        # Gaussian per pixel per view at a mean opacity of 0.223 -- thin, because it
        # assumes ~n/kept of them stacked. Keeping one per voxel and compensating only its
        # area leaves the surface exactly as thin as a single layer, which is why walls
        # came out translucent: 0.223 alone composites to 0.89 accumulated, so a fifth of
        # every wall was see-through. k layers of alpha a compose to 1-(1-a)^k, so that is
        # what the survivor has to carry. At k~10 it takes 0.223 to 0.92.
        if merge == "voxel":
            # exact per-voxel transmittance: what the removed splats composited to
            a_ord = opac[order].clamp(0.0, 1.0 - 1e-6)
            acc = torch.zeros(int(inv.max()) + 1, device=a_ord.device, dtype=torch.float32)
            acc.scatter_add_(0, inv, torch.log1p(-a_ord))
            sel = first[first < n]
            out[4] = (1.0 - torch.exp(acc[inv[sel]])).clamp(0, 1).reshape(-1, 1)
        elif merge == "global":
            k = float(n / kept)
            out[4] = (1.0 - (1.0 - out[4].clamp(0, 1)) ** k).clamp(0, 1)
    return out, v


def surface_distance(maze, pts, tc, tr, span, step=0.03):
    """Distance from each point to the nearest real surface, in metres.

    The labyrinth is axis-aligned, so this decomposes: floor and ceiling are planes, and
    the walls are vertical prisms whose horizontal distance field is one 2D Euclidean
    distance transform of the occupancy grid. No KD-tree, no per-point search.
    """
    from scipy.ndimage import distance_transform_edt
    ox, oz = tile_origin_m(tc, tr, span)
    hi = float(WALL_H * CELL_M)
    half = CELL_M * span + 1.0
    n = int(2 * half / step)
    xs = np.linspace(-half, half, n) + ox
    zs = np.linspace(-half, half, n) + oz
    # solid where the occupancy grid says so: world metres -> grid index
    gi = np.floor(xs / CELL_M).astype(int)
    gj = np.floor(zs / CELL_M).astype(int)
    c0, r0 = gi.min(), gj.min()
    occ = maze.occupancy((c0 - 1) // 2, (r0 - 1) // 2,
                         (gi.max() - c0) // 2 + 3, (gj.max() - r0) // 2 + 3)
    ii = np.clip(gi - ((c0 - 1) // 2) * 2, 0, occ.shape[1] - 1)
    jj = np.clip(gj - ((r0 - 1) // 2) * 2, 0, occ.shape[0] - 1)
    solid = occ[np.ix_(jj, ii)]
    d_out = distance_transform_edt(~solid) * step      # open -> nearest solid
    d_in = distance_transform_edt(solid) * step        # solid -> nearest open
    dw = np.where(solid, d_in, d_out)
    dwt = torch.from_numpy(dw).float().to(pts.device)
    px = ((pts[:, 0] + ox - xs[0]) / step).clamp(0, n - 1).long()
    pz = ((pts[:, 2] + oz - zs[0]) / step).clamp(0, n - 1).long()
    d_wall = dwt[pz, px]
    y = pts[:, 1]
    return torch.minimum(torch.minimum(y.abs(), (hi - y).abs()), d_wall)


def crop_to_footprint(g, span, margin=0.35, ret_mask=False):
    """One cell of geometry per cell of world.

    A tile reconstructs several metres past its own edge, so without this every tile
    holds a second, slightly differently scaled copy of its neighbours' corridors --
    the player ends up standing inside a neighbour's wall. In tile frame the floor is
    at y=0 and the ceiling at WALL_H*CELL_M, so the vertical crop is one storey with a
    little slack rather than a band around the eye."""
    half = float(CELL_M) * span + margin
    m = g[0]
    keep = ((m[:, 0].abs() < half) & (m[:, 2].abs() < half)
            & (m[:, 1] > -0.35) & (m[:, 1] < float(WALL_H * CELL_M) + 0.35))
    out = [t[keep] for t in g]
    if ret_mask:
        return out, int((~keep).sum()), keep
    return out, int((~keep).sum())


def pack_splats(g):
    """[x y z | sx sy sz | qw qx qy qz | r g b | a] float32 interleaved, 14 floats.

    The quaternion is scalar-first. This is a straight dump of the model's rotation head,
    which the official exporter writes into rot_0..rot_3 of a 3DGS PLY, and rot_0 is the
    scalar there. Both this docstring and the client's said `qx qy qz qw`; the shader and
    the measured convention (recon.calibrate_convention picks "wxyz", i.e. no permutation
    before gsplat, which is scalar-first) say otherwise. Reading it as vector-first spins
    every splat by a quarter turn about a mixed axis."""
    buf = torch.cat(list(g), dim=1).contiguous()
    return buf.detach().cpu().numpy().astype("<f4").tobytes(), int(buf.shape[0])


def postprocess(pred, spec, extr, intr=None, depths=None, maze=None, tc=0, tr=0):
    """Predictions -> splats in tile frame: metric, cropped to footprint, pruned.

    Tile frame has the floor at y=0 and the tile centre at x=z=0, so the client places
    a tile by adding tile_origin_m and nothing else.

    floor_err / ceil_err are free validation: the alignment is solved purely from camera
    geometry, so the labyrinth's known storey height is never used to fit anything and
    landing on it is evidence the fit is right.
    """
    g = list(gaussians_from_predictions(pred))
    n_raw = g[0].shape[0]
    conf_all = None
    if "gs_depth_conf" in pred:
        c = pred["gs_depth_conf"][0].float().reshape(-1)
        if c.numel() == n_raw:
            conf_all = c
    if getattr(spec, "unproject", "pred") == "gt" and intr is not None:
        # scale only; positions come from the rig itself
        _, s, rmse = align_prediction([t.clone() for t in g], pred, extr)
        mode = getattr(spec, "scale_from", "camera")
        if mode in ("truth", "surface") and depths:
            st = scale_from_true_depth(pred, depths)
            if st is not None:
                s = st
        if mode == "perview" and depths:
            sv = scale_per_view(pred, depths, int(extr.shape[1]))
            if sv is not None:
                s = sv
        if mode == "surface" and maze is not None:
            try:
                s = refine_scale_on_surfaces(pred, extr, intr, maze, tc, tr, spec.span, s)
            except Exception:
                pass
        g[0] = means_from_true_cameras(pred, extr, intr, s)
        g[1] = g[1] * (float(s.median()) if torch.is_tensor(s) and s.numel() > 1 else s)
    else:
        g, s, rmse = align_prediction(g, pred, extr)
    # AFTER the means are settled, never before: means_from_true_cameras rebuilds one
    # position per pixel from gs_depth, so filtering first leaves the means full length
    # while every other tensor is short, and the next mask indexes a shape that no
    # longer matches.
    if getattr(spec, "conf_percent", 0.0) > 0 and "gs_depth_conf" in pred:
        conf = pred["gs_depth_conf"][0].float().reshape(-1)
        if conf.numel() == g[0].shape[0]:
            keep = conf > quantile(conf, spec.conf_percent / 100.0)
            if int(keep.sum()) > 1000:
                g = [t[keep] for t in g]
    y = g[0][:, 1]
    floor_y = float(quantile(y, 0.02))
    ceil_y = float(quantile(y, 0.98))
    g, n_crop, crop_keep = crop_to_footprint(g, spec.span, ret_mask=True)
    if conf_all is not None:
        conf_all = conf_all[crop_keep]
    n_float = 0
    if getattr(spec, "surface_reject_cm", 0) > 0 and maze is not None:
        d = surface_distance(maze, g[0], tc, tr, spec.span)
        keep = d <= spec.surface_reject_cm / 100.0
        if int(keep.sum()) > 1000:
            n_float = int((~keep).sum())
            g = [t[keep] for t in g]
            if conf_all is not None:
                conf_all = conf_all[keep]
    g, n_big, sq_keep = scale_quantile_filter(g, spec.scale_quantile, ret_mask=True)
    if conf_all is not None:
        conf_all = conf_all[sq_keep]
    g, vox = voxel_prune(g, spec.max_splats if spec.max_splats else g[0].shape[0],
                         merge=getattr(spec, 'opacity_merge', 'global'),
                         grow=getattr(spec, 'scale_grow', True),
                         pick=getattr(spec, 'voxel_pick', 'opacity'),
                         conf=conf_all)
    if spec.opacity_gain != 1.0:
        g[4] = (g[4] * spec.opacity_gain).clamp(0, 1)
    return g, {"raw": n_raw, "cropped": n_crop, "oversized": n_big, "floaters": n_float,
               "voxel": round(vox, 4), "metric_scale": round(float(s.median()) if torch.is_tensor(s) and s.numel() > 1 else s, 4),
               "align_rmse_m": round(rmse, 4),
               "floor_err": round(floor_y, 3),
               "ceil_err": round(ceil_y - float(WALL_H * CELL_M), 3),
               "splats": int(g[0].shape[0])}
