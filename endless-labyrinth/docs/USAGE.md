# Usage

## Requirements

This repository is the labyrinth only. It runs on top of
[HunyuanWorld 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) and its
WorldMirror weights, which you need to obtain separately.

- Python 3.11 with `torch`, `numpy`, `scipy` and the HY-World-2.0 environment
- HY-World-2.0 checked out, with `weights/HY-WorldMirror-2.0` downloaded
- A CUDA GPU
- Any modern browser with WebGL2

## Paths you must set

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
