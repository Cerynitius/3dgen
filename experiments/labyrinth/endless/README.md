# Endless 3DGS labyrinth

> **2026-08-22 现状以 `playground/labyrinth/STATUS.md` 为准。** 本文的迷宫算法、
> 坐标约定、对齐方法、测量方法论仍然有效；"The measured config" 一节及
> CONFIG.md 是历史测量记录——当前配方（官方光学 120°/476 方图、平视环+直下+角落机位、
> 洞穴纹理+低位火盆、NaN 防线、闲时预烘）见 STATUS.md。


An infinite, torchlit labyrinth that is reconstructed into Gaussian splats as the player
walks, using HY-WorldMirror-2.0 feed-forward. Nothing is baked ahead of time and nothing
is stored: every tile is a pure function of `(seed, tile index, spec)`, so walking back
returns the tile you left, byte for byte.

## Why the old code could not be extended

`labyrinth_world.py` was finite by construction, not by configuration. `Cells(cols, rows,
seed)` sizes its wall array by the world; `_repair()` floods the whole grid; `_torches()`
sorts every candidate globally against every other. Each of those needs the entire world
resident before one cell is correct, so "endless" was not a parameter to raise — it
needed a different algorithm.

## Structure

| file | role |
|---|---|
| `infinite_maze.py` | the labyrinth over all of Z², one chunk at a time |
| `render.py` | raycast through a sliding window; world-anchored textures |
| `tiles.py` | tile lattice, camera rig, metric alignment, pruning, packing |
| `recon.py` | WorldMirror forward pass + gsplat scoring against the raycaster |
| `stream_server.py` | ring scheduler, LRU cache, HTTP streaming |
| `bench_config.py` | picks the config by measurement |
| `selftest_maze.py` / `mutation_test.sh` | maze correctness, and proof the tests bite |
| `walk_test.py` | end-to-end: cost stays flat, tiles stay identical |

## How infinity is kept local

Three ideas do all the work, and each is checked in `selftest_maze.py`:

- **Chunks with hashed border gates.** A chunk is `K×K` cells carved by a seeded
  backtracker, so it is internally connected. The openings on a shared chunk edge are
  derived from `hash(seed, edge_id)` alone, so both chunks compute an identical gate set
  without ever meeting. Connectivity over the plane then follows by construction: every
  chunk is connected inside, every adjacent pair shares at least one gate, so the chunk
  adjacency graph is the full grid graph on Z².
- **Torches by local maximum.** A torch is accepted where its hash beats every candidate
  within the spacing radius — a Poisson-disk rule needing only a bounded neighbourhood.
  Placement therefore cannot depend on which box you asked for, and a torch never moves
  when you walk back.
- **Windows with a solid skirt.** The renderer walks a cell box rasterised on demand.
  Its outer ring stays solid: a ray that escapes the array hits nothing, and nothing has
  no surface to shade, so it returns as a black hole mid-corridor.

Textures are keyed on **world** position. Keying them on the window makes brick courses
slide under the player whenever the window re-centres, and puts different mortar on each
side of every tile seam.

## Placing a tile in the world

This is where the reconstruction is easy to get wrong, because every failure looks like
a bad model rather than a bad frame.

The splat means are unprojected from `gs_depth` using the model's own **predicted**
cameras (`position_from="gsdepth+predcamera"`). Those live in a frame the network
normalises for itself — conditioning on poses steers it, it does not pin it. Measured on
one tile, the predicted rig spanned `1.29 × 0.05 × 1.28` where the real rig spans
`8.65 × 0 × 8.62`: same shape, ~6.7× smaller. Assuming the output was already in camera
0's frame put every tile in the wrong place at the wrong size, and nothing downstream
could have corrected it.

Since the predicted cameras are returned with the predictions, the transform is solved,
not guessed: `align_prediction()` fits the similarity carrying predicted camera centres
onto the true ones (Umeyama, with the reflection guard) and applies it to the Gaussians,
rotating each quaternion by the same rotation.

The fit uses camera geometry only. That makes the labyrinth's known storey height a free
check rather than an input — and the floor lands within a few centimetres of `y=0` with
the ceiling within centimetres of 3.20 m. Both are reported per tile as `floor_err` /
`ceil_err`, alongside `align_rmse_m`, so a tile that failed to align can be detected
instead of served.

## Conventions, settled by measurement

All of these are recorded because getting one wrong costs quality silently rather than
crashing:

- **Cameras are OpenCV** — `[right, down, forward]`, det +1. The natural
  `[right, up, forward]` is a *left-handed* camera basis: it renders vertically mirrored
  and feeds the model poses that disagree with their images. `render.camera_basis()` is
  the single definition, shared by the raycaster, the pose priors and gsplat.
- **Scales are linear metres** (`reg_dense_scales` is `exp`, and the exporter writes
  `scales.log()`).
- **Opacities are already probabilities** (`reg_dense_opacities` is `sigmoid`), and are
  used as-is. A second sigmoid scores ~7 dB better in calibration, which looked like
  evidence that the field wanted softening — the head emits one Gaussian per pixel per
  view, so every surface is represented `n_pos*n_az` times over. It isn't. On values
  already in `[0.14, 0.99]` a second sigmoid maps to `[0.54, 0.73]`, *raising* the
  transparent splats as much as it lowers the opaque ones. Sweeping a true softening
  (`opacity_gain`) is monotonically worse — 20.28 dB at 1.0 against 18.41 at 0.3 — so the
  7 dB came from range compression on a rig view, not from reducing opacity.
- **Quaternions are wxyz** (the exporter writes them straight into `rot_0..rot_3`).

## The measured config

`span=3, n_pos=4, n_az=6, size=504, max_splats=1_500_000` — 1.72 s/tile, ~1.17M splats,
62.6 MB/tile, 17.0 cm median depth error, sustains 2.3 m/s of walking.

Everything here was swept, not chosen:

| axis | measured |
|---|---|
| rig | quality rises with views (1×8 → 9×6 gives 15.1 → 24.1 dB) but 9×6 sustains only 0.87 m/s. 4×6 is the fastest rig that outruns a walk |
| resolution | 364/504/644/952 → 19.9/20.3/20.7/22.0 dB at 1.1/1.7/2.7/7.5 s. 952 drops walking speed to 0.53 m/s for 2 dB |
| span | tile cost is flat (~1.72 s) since the rig is a fixed 24 views, so span trades coverage against tiles-per-metre. span 3 has the best coverage (83.8%) and clears walking pace at 2.3 m/s |
| opacity | gain 1.0 wins (20.28 dB) and every softening step is worse. The model's own opacities are correct |
| budget | **1.17M splats beats the unpruned 4.1M by 6.7× on depth error** (17.0 cm vs 113.8 cm) |

That last row is the one to keep. The splat head emits one Gaussian per pixel per view,
so the raw field is redundant by `n_pos*n_az` and the duplicates z-fight rather than
reinforce. Voxel pruning is not a quality sacrifice made for bandwidth here — it is what
makes the geometry correct, and the 3.5× smaller payload is a side effect.

### Why the metric is depth, not PSNR

Colour PSNR is actively misleading on this world and was nearly the basis of the wrong
recommendation. A torchlit vault averages ~0.09 brightness, so a flat dark frame
reproduces it to within a few percent. Scored on colour, quality appeared to rise
monotonically as tiles were pruned to nothing — 26.8 dB at 19k splats, better than any
richer tile — while measured edge energy fell to 0.00006 against the ground truth's
0.00053. It was buying PSNR with blur.

Scored on **depth**, which darkness cannot flatter, the ranking inverts completely:

| splats | depth MAE | colour PSNR |
|---|---|---|
| 4,117,722 | 113.8 cm | 20.28 dB |
| **1,171,991** | **17.0 cm** | 22.76 dB |
| 333,410 | 23.7 cm | 24.66 dB |
| 19,445 | 66.9 cm | 26.78 dB |

Colour PSNR is monotonically *anti-correlated* with geometric accuracy across this whole
range. It is still reported, because it catches convention errors that depth is blind to
— but it does not choose the config.

Honest limits: only ~27% of pixels land within 10 cm, and ~20% of pixels the raycaster
sees have no splat coverage at all. This is a 1.7 s feed-forward pass, not the 41-minute
offline path.

## Scoring

The raycaster is the ground truth — it drew the views the model was given, so any
disagreement is the reconstruction's. Two rules learned the hard way:

- Score a **block** of assembled tiles, never one tile. A camera inside a lone tile sees
  past its crop into territory that is legitimately empty, so three quarters of the frame
  is black and the number means nothing.
- Settle **conventions** against a rig view, not a novel one. At a held-out pose every
  candidate is wrong somewhere, and on a scene this dark PSNR is then decided by mean
  brightness — which is how a camera pointing backwards came out ahead twice.

`bench_config.py` reports interior and seam PSNR separately, because a tile that
reconstructs beautifully alone is worthless if it does not meet its neighbour.

## Running it

```bash
conda activate hyworld2
python selftest_maze.py          # maze invariants, CPU only
bash   mutation_test.sh          # proof those tests actually bite
python bench_config.py --stage all
python stream_server.py --port 8092 --seed 1000
python walk_test.py --steps 70 --speed 1.4   # end-to-end, at walking pace
```

`walk_test.py` follows a BFS route to a distant goal rather than wandering — a
straight-biased random walk covered 12 cells in 200 steps, which tests the cache and
nothing else — and lets real time pass between steps at `--speed` m/s. Without that it
is measuring an infinitely fast player, who outruns any config.

Measured, walking 71 cells (~284 m) at 1.4 m/s, uncontended:

```
cold start: opening ring of 9 tiles in 15.3s
walked 71 cells across 22 tiles, built 69 in 206s  (mean 1.89s/tile)
  PASS  every tile the player stood in was delivered            22 of 22
  PASS  cache stayed inside its bound                           3046.0MB of 3072MB
  PASS  a rebuilt tile is byte-identical to the first time      tile (0, 0)
  PASS  a tile 48 km out costs the same as one at the origin    1.90s vs mean 1.89s
  PASS  the walker is not outrun by the generator               0 stalls, 0.0% of 206s
  PASS  sustained build rate keeps ahead of the walk            5.7s per crossing vs 8.6s
GPU peak 15.8 GB
```

Cold start is reported separately on purpose: the opening ring fills serially, so a
player who starts walking the instant the server boots waits ~15 s for reasons that say
nothing about steady-state throughput. After that the generator stays ahead with ~35%
headroom, and the tile 48 km from the origin costs what the first one did.

Client contract: splat means are **tile-relative**; add `origin_m` from the tile's
metadata to place it. Stride is 14 float32:
`[x y z | sx sy sz | qw qx qy qz | r g b | a]`. `/maze` serves wall bits so the client
can collide locally without a round trip.

### The browser client has not been migrated

`playground/labyrinth/live.js` still points at the old server — `:8091` and `/chunk`,
addressed by **cell**, with the world's size fixed at startup. This server is addressed
by **tile** on an unbounded lattice and expects the client to drive generation from the
player's position (`/ring`), so the two are not drop-in compatible. Porting it means
three changes: ask `/ring` each frame instead of requesting a fixed cell grid, place
tiles by `origin_m` rather than by the old per-cell offset, and drop tiles that leave the
ring. Nothing in the JS has been touched.

## Relationship to the official worldgen path

`run_worldgen.sh` (WorldNav → WorldStereo → `gen_gs_data` → `world_gs_trainer`) costs
~41 min per world and cannot feed a walk. It is not in this loop. What is carried over
from it is its settled behaviour: 14-px-snapped view sizes, the 0.98 max-scale quantile
filter from `_build_gs_ply_data()`, and voxel — never random — thinning, so that
coverage stays uniform at any budget instead of losing surfaces in clumps.
