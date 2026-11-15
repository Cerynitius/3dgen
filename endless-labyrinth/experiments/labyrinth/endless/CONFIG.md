# The default config

Every value below is a default in the code (`TileSpec` in `tiles.py`, the lighting
constants in `render.py`), so `python stream_server.py --warm 1` with no flags *is* this
config. Each value is here because a measurement chose it; re-run the sweep that owns a
value before changing it.

## Tile and rig

**`span = 2`** — 2×2 cells per tile, and the rig places a camera in every cell by
construction. With fewer positions than cells, unobserved cells that were also walled
off from the observed ones came back with literally zero splats — holes in the walls.

**`n_pos = 4, n_az = 4` + nadir + corner cameras** — at each cell: a level ring of 4
azimuths stepped `baseline = 0.35` m off the centre, one downward-pitched view from the
centre (−0.45 rad — the level ring never sees the floor head-on), and one view pitched
−0.5 rad into each recessed corner (two walls meeting form a blind pocket for a
centre-standing rig). The view count therefore varies with the maze topology, roughly
32–48 per tile.

**`n_context = 4`, `n_az_context = 3`** — cameras in the ring of cells *outside* the
tile, looking back in, chosen by bearing (uniform in direction, not in list order) with
visibility preferred. A tile's boundary strip is reconstructed almost entirely from the
context camera facing it; every chunked 3DGS system in the literature gives a chunk
more input than its own footprint and crops the output back. Against the best
no-context rig this is better on every axis *and* faster on fewer views: depth accuracy
23.1 → 14.6 cm, splats within 5 cm of a real surface 19.7 → 33.6%, true surface covered
43.1 → 61.2%.

**`size = 476`, `aspect = 1.0`** — the official 480 px square, snapped to the ViT's
14 px patches. The official path feeds square input, so square is what the network has
seen.

**`fov_x = 120°`** — the wider vertical field is what fixed the empty floor and
ceiling bands at no extra pixels: frame with no geometry 14.7% → 10.0%, floor holes
30% → 18%, and depth error *improves* 14.4 → 12.1 cm. The cost is angular resolution
(edge energy 0.99× → 0.82× of ground truth).

## Metric frame

**`unproject = "gt"`** — splat positions come from `gs_depth` unprojected with the
rig's exact poses, not the model's predicted cameras (which invent ~31 cm of vertical
spread on a rig that is level by construction). Depth error falls 31.7 → 23.8 cm at no
cost.

**`scale_from = "perview"`** — one metric scale per view, fitted against that view's
own raycast depth. Against one global scale: F@5cm 31.1 → 40.2%, worst tile
17.5 → 32.9%, floor error 5.6 → 3.2 cm. The worst-tile number is the point: a single
global scale left some tiles badly stretched, and a stretched tile reads on screen as
scatter.

**`surface_reject_cm = 10`** — the labyrinth is ours, so a splat further than 10 cm
from any real surface is provably wrong and dropped. 10 cm, not tighter: the rejection
threshold and the audit tolerance are the same operation, so tightening it would let
the filter grade itself. The gain is the fixed budget being spent on real surface
(coverage 60.8 → 62.0%, ~160k floaters removed per tile).

## Pruning

**`max_splats = 400_000`** (voxel-prunes to ~340k, ~18 MB on the wire) — pruning is
not a bandwidth compromise. The splat head emits one Gaussian per pixel per view, so
the raw field is redundant by the view count and the duplicates z-fight rather than
reinforce: unpruned tiles measure 113.8 cm median depth error against ~17 cm pruned.
Thinning is by voxel, never at random, so coverage stays uniform at any budget.

**`scale_quantile = 0.98`** — the official PLY writer's filter, dropping Gaussians
stretched across depth discontinuities. Disabling it takes depth error to 200.6 cm;
relaxing it to 0.995 fills holes by smearing (edge energy 0.99× → 0.71×), and blur
degrades every pixel where a hole degrades one.

**`opacity_merge = "global"`, `scale_grow = True`** — the voxel survivor carries what
the removed splats composited to: opacity `1-(1-a)^k` and radius grown `√k`. Without
the opacity compensation walls render translucent; without the growth the voxel grid
shows through as a lattice of gaps (see-through 16.6% → 26.1%).

**`opacity_gain = 1.0`** — swept, and 1.0 won against every softening step. The
model's own opacities are correct.

## Lighting

The active model in `render.shade` is the cave variant: a flat diffuse base per
surface (walls 0.45, floor 1.05, ceiling 0.40 — the floor is the player's ground truth,
the ceiling is deliberately dark), plus the 8 nearest braziers with the flame at
0.55 m shining upward, a soft `1/(1+(0.5d)²)·exp(−(d/14)²)` falloff, and a light pool
hugging the floor line on walls. Its constants are inline in `shade`; the sweep target
was no view below the model's signal floor while contrast still reads as torchlight.
`LIGHT_CAP = 48` culls the window's torch list per view. The module-level
`REACH_M` / `INTENSITY` / `FALLOFF_K` / `AMBIENT` / `FILL` constants are not used by
this path — they persist because the world `version` hashes them.

## Client

**20 slots × 400k = 8M splats** — a slot must hold the largest tile the server can
emit (`voxel_prune` accepts up to 1.1× its target), and a radius-2 ring wants 25 tiles,
so fewer, larger slots churn on every step.

**`dilate = 0.30`, `gain = 1.0`** — `dilate` is gsplat's `eps2d` (the same 2D
covariance dilation), swept offline by `bench_client.py` on exposure ratio and edge
energy. `gain` is 1.0 because opacity is already compensated server-side at voxel
prune; it remains as a query-parameter escape hatch.

## Changing it

Re-run the sweep that owns the value: `bench_config.py` for the rig/resolution/budget,
`bench_light.py` for lighting, `bench_client.py` for the shader constants. Each prints
the whole curve, not just a winner. The world `version` in `/spec` hashes seed, spec,
torch spacing, lighting and the tile-generating source, so a changed world can never be
served from a browser's year-long tile cache.
