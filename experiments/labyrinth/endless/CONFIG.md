# The examined config

Every value below is a default in the code, so `python stream_server.py --warm 1` with no
flags *is* this config. Each one is here because a measurement chose it, and the
measurement is named. Where something was picked by judgement rather than by number, it
says so.

```
[server] seed=1000 spec=s2_4x6_p504_m400000
[server] tile 8x8 m, 4x6=24 views at 504px, budget 400,000 splats
[server] torch every 5.0 m, falloff k=0.45, fill=0.15, reach=22.0 m
[server] world version e1e79cfd1aff
```

## Measured behaviour of this config

| | measured | how |
|---|---|---|
| coverage | 0 starved cells of 24, min **60,699** splats/cell | splat count per cell over 6 tiles |
| geometry | **14.4 cm** median depth error | vs raycast depth, 8 cameras on a 2×2 block |
| fill | 79.4% of ground-truth pixels have splats | same cameras |
| brightness | median **0.449**, p90 0.782, p01 0.171 | 16 views over 4 cells |
| throughput | **1.77 s/tile**, sustains **1.50 m/s** | 3 cold tiles, ring radius 1 |
| determinism | rebuilt tile byte-identical | blake2b of the packed payload |
| wire | 16.0 MB/tile | served response |

Suites: `selftest_maze.py` ALL PASS · `mutation_test.sh` ALL MUTATIONS CAUGHT ·
`endless_selftest.mjs` ALL PASS.

## Why each value

**`span = 2`** — the rig places `n_pos` cameras among `span²` cells. At span 3 with
`n_pos=4` only four of nine cells were ever observed, and cells with no camera that were
also walled off from the ones that had one came back with *zero* splats:

```
cell 0 CAM  63,671      cell 3 CAM  70,698      cell 6      0
cell 1      33,678      cell 4      49,330      cell 7      0
cell 2       1,522      cell 5 CAM  47,449      cell 8 CAM 63,791
```

Those were the missing walls. At span 2 every cell has a camera by construction. The
alternatives that also close the gap are worse:

| config | starved | min/cell | s/tile | sustains |
|---|---|---|---|---|
| span3 4×6 | 6/27 | 0 | 1.85 | 2.16 m/s |
| span3 9×4 | 0/27 | 7,349 | 2.82 | 1.42 m/s |
| span3 9×6 | 0/27 | 5,800 | 4.68 | 0.85 m/s |
| **span2 4×6** | **0/12** | **63,380** | 1.80 | 1.49 m/s |

**`n_az = 6`, `size = 504`** — swept together against speed and geometry at span 2:

| | starved | s/tile | sustains | depth MAE | filled |
|---|---|---|---|---|---|
| 4az 504px | 0/16 | 1.22 | 2.19 m/s | 46.9 cm | 78.3% |
| 6az 420px | 0/16 | 1.34 | 1.99 m/s | 23.9 cm | 83.3% |
| **6az 504px** | 0/16 | 1.78 | **1.50 m/s** | **16.6 cm** | 85.3% |
| 6az 588px | 0/16 | 2.32 | 1.15 m/s | 13.3 cm | 87.3% |

588px is better geometry and cannot keep ahead of a walk. 420px is the fallback if
stalling matters more than fidelity.

**`max_splats = 400_000`** (voxel-prunes to ~340k) — pruning is not a bandwidth
compromise. Unpruned tiles measure **113.8 cm** median depth error against 17.0 cm at
1.17M and 23.7 cm at 333k: the splat head emits one Gaussian per pixel per view, so the
raw field is redundant by `n_pos*n_az` and the duplicates z-fight rather than reinforce.
400k is the point on that curve that also fits the browser renderer's slot.

**Torch spacing 1.25 cells (5 m), `FALLOFF_K = 0.45`, `FILL = 0.15`** — swept by
`bench_light.py` against a stated target, not to taste. Before: median 0.093, nothing
above 0.21, five of sixteen views unnavigable. After: median 0.449, darkest view 0.235,
none below 0.20, p90/p10 contrast 3.6× so torchlight still reads as torchlight.

**`opacity_gain = 1.0`** — swept, and 1.0 won against 0.8/0.6/0.45/0.3 (20.28 / 19.89 /
19.41 / 18.97 / 18.41 dB). The model's own opacities are correct.

**Camera convention `none` (OpenCV), quats `wxyz`, colours SH-DC, opacities raw** —
settled against a rig view the model was actually given, where the correct convention has
to reproduce the exact input image. `none` won at 32.6 dB with 99.8% coverage.

**Client: 18 slots, `gain=1.8`, `dilate=0.30`, `radius=1`** — 18 because a slot must hold
the largest tile the server can emit, and `voxel_prune` accepts up to 1.1× its target, so
a 400k budget can return ~440k. At 24 slots each holds 333,312 and every tile is silently
truncated.

`gain` and `dilate` live in the WebGL shader, which is why they went unswept — with no
browser there was nothing to score. Both have exact offline equivalents, so
`bench_client.py` examines them anyway: `dilate` *is* gsplat's `eps2d` (the same 2D
covariance dilation, and gsplat's default `rasterize_mode="classic"` leaves opacity
uncompensated exactly as the shader does), and `gain` is a pure opacity remap applied
before projection. Scored on exposure ratio and edge energy, not PSNR alone — PSNR
chose blur once already.

| gain | dilate | exposure | edge |
|---|---|---|---|
| 1.4 | 0.30 | 0.78× | 1.07× |
| **1.8** | **0.30** | **0.79×** | **0.99×** |
| 2.4 | 0.30 | 0.81× | 0.92× |
| 3.2 | 0.30 | 0.81× | 0.87× |

Above ~1.8 gain buys almost no exposure and steadily smears detail. What settles it: on
pixels that *have* splats, exposure is 0.97× at gain 1.0, **1.01× at 1.8**, 1.02× at 3.2.
1.8 is where the field is correctly exposed.

## Not examined

Stated plainly so it is not mistaken for settled:

- **Nothing has been visually verified in the browser.** Every number here is headless:
  gsplat renders, depth against the raycaster, Node against the server. The WebGL path
  shares the wire format and the placement arithmetic, not the code.
- **`fov_x`, `baseline`, `view_margin`** were inherited or set once and never swept.
  (`pitch_amp` and `scale_quantile` have since been swept — see the two entries below.)
- **14.7% of the frame has no geometry at all**, and it is *not* an artefact of scoring:
  99.2% of unfilled pixels are inside the block, not beyond it. They cluster on the floor
  (30%) and ceiling (25%) rather than the walls (~11%). Raising the rig's pitch makes it
  worse, not better (fill 79% → 65%, depth 14.4 → 31 cm), so it is not simply that those
  surfaces go unobserved. Unresolved.
- **Relaxing `scale_quantile` fixes coverage and is refused anyway.** 0.995 takes empty
  from 14.7% to 10.6% at +0.2 cm depth — but the splats it readmits fill those holes by
  smearing, dropping edge energy from 0.99× of ground truth to 0.71×. Blur degrades every
  pixel; a hole degrades one. Recorded because it is the most tempting wrong turn here.
- **Sprint outruns the generator.** 1.50 m/s sustained against a 1.4 m/s walk is 7%
  margin; Shift is 2.6 m/s.
- **`seed`, `k=8` chunk size, `CHUNK`/`braid`/`room_tries`** are structural choices, not
  tuned ones.

## Changing it

Re-run the sweep that owns the value: `bench_config.py` for the rig/resolution/budget,
`bench_light.py` for lighting. Both print the whole curve, not just a winner. The world
`version` in `/spec` hashes seed, spec, torch spacing and every lighting constant, so a
changed world cannot be served from a browser's year-long tile cache.
