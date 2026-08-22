# Endless 3DGS labyrinth

An infinite, torchlit labyrinth that is reconstructed into Gaussian splats as the player
walks, using HY-WorldMirror-2.0 feed-forward. Nothing is baked ahead of time and nothing
is stored: every tile is a pure function of `(seed, tile index, spec)`, so walking back
returns the tile you left, byte for byte.

## Structure

| file | role |
|---|---|
| `infinite_maze.py` | the labyrinth over all of Z², one chunk at a time |
| `render.py` | raycast through a sliding window; world-anchored textures |
| `tiles.py` | tile lattice, camera rig, metric alignment, pruning, packing |
| `recon.py` | WorldMirror forward pass + gsplat scoring against the raycaster |
| `stream_server.py` | ring scheduler, LRU cache, HTTP streaming |
| `bench_config.py`, `bench_light.py`, `bench_seams.py`, `bench_surface.py`, `bench_client.py` | the measurement suites that choose the config |
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
  no surface to shade, so it would return as a black hole mid-corridor.

Textures are keyed on **world** position, never on the window, so surface detail does
not slide under the player when the window re-centres and both sides of a tile seam
carry the same pattern.

## Placing a tile in the world

A tile's metric origin is pure arithmetic on its lattice index, so the client places it
by translation alone. Inside the tile:

- Splat positions are unprojected from the model's `gs_depth` using the rig's **own
  poses** (`unproject="gt"`), which are exact by construction — the model's predicted
  cameras never enter the points.
- The model's depth is in its own normalised units, so each view's metric scale is
  fitted against the raycaster's true depth of that same view
  (`scale_from="perview"`) — one robust median ratio per view.
- The reconstruction extends past the tile's own edge, so it is cropped to the tile's
  footprint: one cell of geometry per cell of world.
- The labyrinth is ours, so a splat further than 10 cm from any real surface is provably
  wrong and rejected; the freed budget is spent on real surface.
- The 0.98 max-scale quantile filter drops Gaussians stretched across depth
  discontinuities, and voxel pruning thins to the splat budget — with the survivor's
  opacity compensated `1-(1-a)^k` and its radius grown by `√k`, so the thinned field
  stays as opaque and gap-free as the full one.

The similarity fit between predicted and true camera centres (Umeyama, reflection
guard) is still solved per tile: its residual `align_rmse_m`, and `floor_err` /
`ceil_err` against the labyrinth's known storey height, are served in the tile metadata
as validation that costs nothing — none of them is an input to the placement.

## Conventions

Getting any of these wrong costs quality silently rather than crashing, so they are
recorded and covered by `recon.calibrate_convention`:

- **Cameras are OpenCV** — `[right, down, forward]`, det +1. `render.camera_basis()` is
  the single definition, shared by the raycaster, the pose priors and gsplat.
- **Scales are linear metres.**
- **Opacities are probabilities** (`reg_dense_opacities` is `sigmoid`) and are used
  as-is: `opacity_gain = 1.0`, measured best.
- **Quaternions are wxyz**, scalar first, on the wire and in the shader.

## Scoring

The raycaster is the ground truth — it drew the views the model was given, so any
disagreement is the reconstruction's fault. Three rules:

- **Depth chooses configs, not colour PSNR.** A torchlit vault averages ~0.09
  brightness, so colour PSNR rewards blur and near-empty frames; depth cannot be
  flattered by darkness. Colour is still reported because it catches convention errors
  depth is blind to.
- Score a **block** of assembled tiles, never one tile: a camera inside a lone tile sees
  past its crop into territory that is legitimately empty.
- Settle **conventions** against a rig view the model was actually given, where the
  correct convention has to reproduce the exact input image.

`bench_config.py` reports interior and seam quality separately, `bench_surface.py`
audits each tile against the true surfaces, and `bench_seams.py` measures exactly where
neighbouring tiles have to agree.

## Running it

```bash
conda activate hyworld2
python selftest_maze.py          # maze invariants, CPU only
bash   mutation_test.sh          # proof those tests actually bite
python stream_server.py --port 8092 --seed 1000 --warm 1
python walk_test.py --steps 70 --speed 1.4   # end-to-end, at walking pace
```

`walk_test.py` follows a BFS route to a distant goal rather than wandering, and lets
real time pass between steps at `--speed` m/s, so it measures a real walker rather than
an infinitely fast one. It asserts that every tile the player stood in was delivered,
the cache stays inside its bound, a rebuilt tile is byte-identical, and a tile 48 km
out costs the same as one at the origin.

Client contract: splat means are **tile-relative**; add `origin_m` from the tile's
metadata to place it. Stride is 14 float32:
`[x y z | sx sy sz | qw qx qy qz | r g b | a]`. `/maze` serves wall bits so the client
can collide locally without a round trip.

The default configuration and the measurements behind each value are in `CONFIG.md`.
