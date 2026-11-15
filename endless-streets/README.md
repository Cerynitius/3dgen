# Endless Streets

An endless generated town you can walk through in the browser. Each part of the town is turned into 3D Gaussians the
first time it is needed, and every frame is cleaned up live by FastFix3D (UltraFix Fast).

It is a practice ground for route training of older adults with MCI or early Alzheimer's disease: landmarks stand at
about half of the crossings, every street keeps its own trees and shop signs, a place you walk back to looks exactly as
you left it, and the ray-traced town is the ground truth that FastFix3D's output can be checked against (key `3`).

The buildings' looks are generated while you walk: every building gets its own front-elevation photo from
Qwen-Image-2.1-Turbo (style, number of floors and a shop with a Chinese sign chosen from the building's seed), so a new
street brings new shops, and an old street keeps its own. The same goes for the asphalt of each street segment, the
paving of each block, the street trees (one species per street), statues, benches, kiosks, poster columns and street
clocks; only the road markings, kerbs and lamp posts are drawn by code.

## Run

```bash
./assetgen.sh       # facade generator on the RTX PRO 4000, 127.0.0.1:8097
./run.sh --build-gpus 1                             # walk + FastFix3D on GPU 0, baked tiles on GPU 1, port 8095
PORT=8096 ./run.sh --recon worldmirror --build-gpus 1   # same with WorldMirror tiles
```

Open `http://<host>:8095/`, click the picture to capture the mouse, walk with `W`/`A`/`S`/`D`, run with `Shift`.

| Key | View |
|---|---|
| `1` | 3DGS render |
| `2` | FastFix3D |
| `3` | ground truth (ray tracer) |
| `4` | 3DGS on the left, FastFix3D on the right |

Options: `--seed` (a different town), `--radius` (tiles loaded around the walker), `--lod` / `--lod-dist` (level
of detail), `--no-fix`, `--port`, `--no-assets` (procedural facades only). Tiles are cached in `cache/`, one directory per seed and setting.

Needs the FastFix3D engine at `runs/fast_960x540.engine` (build it with `scripts/fast/build_engine.py` of
UltraFix, `SIZE=960x540`).

## Files

| File | |
|---|---|
| `world.py` | the town: street grid, buildings, signs, trees, lamps, landmarks at crossings |
| `cityassets.py` | which photo each building needs (key, prompt, size) |
| `assetgen.py` | facade generator service: Qwen-Image-2.1-Turbo FP8, crop to the front wall |
| `raytrace.py` | GPU ray tracer (Triton) with textures, sun and shadows |
| `stiles.py` | tiles: camera rig, splats from the rig views, pruning |
| `splatrender.py` | gsplat rendering |
| `server.py` | tile store, walker, frame pipeline, WebSocket server |
| `client/walk.html` | the page |
| `walk_test.py`, `evaltile.py` | headless walk and tile scoring |
