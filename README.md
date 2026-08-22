# Endless Labyrinth

A first-person labyrinth with no edge, rendered as 3D Gaussian splats that are
reconstructed on demand while you walk.

Nothing is pre-baked. The maze is an infinite procedural lattice, and the geometry you
see is produced by a feed-forward reconstruction model that is handed a rig of
raycast views of the cells you are approaching and returns splats for them. Every tile
is a pure function of `(seed, tile index, spec)`, so a tile you walk back to is the tile
you left — byte for byte — and the cache can be thrown away under pressure without the
world changing.

```
     browser  ──  /spec   metric constants the client places tiles by
                  /maze   wall bits, so collision is resolved locally
                  /ring   which tiles surround you, nearest first
                  /tile   binary splats, 14 floats each
```

## What is in here

| | |
|---|---|
| `playground/labyrinth/` | the client: WebGL2 splat renderer, controls, collision, streaming |
| `experiments/labyrinth/endless/` | the server: maze, raycaster, camera rig, reconstruction, HTTP |

The client is plain ES modules with no build step and no dependencies. The server is
Python and needs a GPU.

### Client

| file | |
|---|---|
| `endless.js` | entry point: streaming loop, ring pump, HUD |
| `splats.js` | WebGL2 splat field — texture-backed slots, LRU eviction, depth sort worker |
| `controls.js` | first-person controller, exact exponential-approach integration |
| `occupancy.js` | odd/even occupancy grid rebuilt from the wall bits `/maze` serves |
| `netpolicy.js` | deadlines, exponential backoff, cost-keyed refusal cache |
| `geometry.js`, `collider3d.js`, `mat4.js`, `maze.js` | mesh building, collision, matrix maths, local generator |
| `endless.html` | the page |

`live.html` / `live.js` are a separate mesh-rendered view of the same maze, kept for
comparison against the splat path.

### Server

| file | |
|---|---|
| `stream_server.py` | HTTP endpoints, tile cache, build queue, worker threads |
| `infinite_maze.py` | the infinite lattice: hashed chunk gates, occupancy rasterisation |
| `render.py` | GPU raycaster — the views the model is conditioned on, and true depth |
| `tiles.py` | tile geometry, camera rig, metric scale, cropping, pruning, wire packing |
| `recon.py` | the reconstruction model wrapper |

## Requirements

This repository is the labyrinth only. It runs on top of
[HunyuanWorld 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) and its
WorldMirror weights, which you need to obtain separately.

- Python 3.11 with `torch`, `numpy`, `scipy` and the HY-World-2.0 environment
- HY-World-2.0 checked out, with `weights/HY-WorldMirror-2.0` downloaded
- A CUDA GPU (the reference machine builds a tile in roughly 2.5–4 s)
- Any modern browser with WebGL2

### Paths you must set

The server locates HY-World-2.0 by absolute path. Point these at your own checkout:

```
experiments/labyrinth/endless/recon.py:28    sys.path.insert(0, ".../HY-World-2.0")
experiments/labyrinth/endless/recon.py:36    WEIGHTS = ".../HY-World-2.0/weights"
experiments/labyrinth/endless/tiles.py:563   sys.path.insert(0, ".../HY-World-2.0")
experiments/labyrinth/endless/tiles.py:648   sys.path.insert(0, ".../HY-World-2.0")
```

The benchmarks and `mutation_test.sh` carry the same path, and
`playground/labyrinth/bugfix_test.mjs` defaults to a specific interpreter which you can
override with `LAB_PYTHON`.

## Running it

Two processes. The tile server holds the model resident; the client is static files.

```bash
# 1. the tile server (GPU)
cd experiments/labyrinth/endless
python stream_server.py --port 8092 --warm 1

# 2. the client, from any static file server
cd playground/labyrinth
python -m http.server 8090
```

Then open `http://<host>:8090/endless.html`.

`--warm 1` builds the spawn ring before the port opens, so the first frame already has
geometry. Without it the world assembles around you as you stand there.

The client finds the tile server at `<same host>:8092` by default. Point it elsewhere
with `?server=`:

```
http://localhost:8090/endless.html?server=http://gpu-box:8092
```

Other query parameters: `radius` (ring radius, 0–4), `gain`, `dilate`. All are
range-checked; an out-of-range or non-numeric value falls back to the default.

Click to capture the pointer, `W`/`A`/`S`/`D` to walk, `Shift` to run.

## Verifying a checkout

```bash
cd playground/labyrinth
node bugfix_test.mjs         # client behaviour, no browser needed
node endless_selftest.mjs    # occupancy vs the server's rasterisation
./mutation_bugfix.sh         # reintroduces each fixed bug and requires the suite to catch it

cd experiments/labyrinth/endless
python selftest_maze.py      # maze determinism, connectivity, chunk borders, conventions
python bench_seams.py        # reconstruction quality at tile joins, against occupancy truth
python bench_surface.py      # per-tile surface audit
```

`bugfix_test.mjs` and `selftest_maze.py` run without a GPU. The benchmarks measure a
running server and take `--server`.

## Design notes

`playground/labyrinth/STATUS.md` is the current state of the pipeline — the camera
rig and optics, the render and forward-pass sanitising, the server and client policies,
and the known ceilings — written in Chinese. `experiments/labyrinth/endless/README.md`
covers the server design, and `experiments/labyrinth/endless/CONFIG.md` records the
default configuration and the measurement behind each value.

## Licence and attribution

The labyrinth code in this repository is its authors'. It depends on, and is designed
around, Tencent's HunyuanWorld 2.0 and the WorldMirror reconstruction model, which carry
their own licence — see the [upstream repository](https://github.com/Tencent-Hunyuan/HY-World-2.0).
No upstream code or model weights are redistributed here.
