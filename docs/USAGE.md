# Usage

## Endless Streets

Run everything from `endless-streets/`.

### Requirements

- **Walk server:** Python with PyTorch, Triton, gsplat, aiohttp, TensorRT and the `ultrafix` package (UltraFix source tree), plus HY-World-2.0 for WorldMirror tiles.
- **FastFix3D engine:** `runs/fast_960x540.engine`, built with UltraFix `scripts/fast/build_engine.py` and `SIZE=960x540`.
- **Image generator:** Python with diffusers (Qwen-Image 2.1 pipeline), torchao, transformers, FastAPI, uvicorn and SciPy, on a 24 GB card; Qwen-Image-2.1-Turbo and SegFormer-B2 Cityscapes weights.

| Where | Default | What it points to |
|---|---|---|
| `run.sh`: `FAM3DGS_ENV` | `~/fam3dgs/env-dev.sh` | script that sets up the CUDA / gsplat environment |
| `run.sh`: `WALK_PY` | `~/fam3dgs/venv-difix/bin/python` | the Python of the walk server |
| `run.sh`: `WALK_CUDA_DEVICES` | `0,1,2` | the cards the walk server may use |
| `server.py`: `--weights` | `~/src/ultrafix/weights/ultrafix_fast.safetensors` | UltraFix Fast weights; the source tree is `~/src/ultrafix/src` |
| `stiles.py` | `~/models/HY-World-2.0` | WorldMirror weights |
| `assetgen.sh`: `IMGGEN_ENV` | `~/miniconda3/envs/imggen21` | the conda env of the image generator |
| `assetgen.sh`: `ASSETGEN_GPU` | the RTX PRO 4000 | the card of the image generator (index or UUID) |
| `assetgen.py` | `~/models/imggen/Qwen-Image-2.1-Turbo`, `~/models/segformer-b2-cityscapes` | generator and facade-crop weights |

### 1. Start the image generator

```bash
./assetgen.sh            # 127.0.0.1:8097; serves once loaded and warmed up
```

### 2. Start the walk server

```bash
./run.sh --build-gpus 0,1                                 # baked tiles, port 8095
PORT=8096 ./run.sh --recon worldmirror --build-gpus 1     # WorldMirror tiles, port 8096
```

| Option | Default | Effect |
|---|---|---|
| `--seed` | `0` | the town and its routes |
| `--walk-gpu` | `0` | card for the walk frames and FastFix3D |
| `--build-gpus` | `1,2` | cards that build tiles |
| `--radius` | `3` | tiles loaded around the walker |
| `--recon` | `bake` | `bake` (splats from ray-traced views) or `worldmirror` |
| `--lod`, `--lod-dist` | `0.025,0.06,0.15`, `8,25` | voxel sizes of the three detail levels, and the distances (m) where they change |
| `--no-fix` | off | no FastFix3D |
| `--no-assets` | off | procedural looks only, no generated images |
| `--no-holefill` | off | no hole patching while walking |
| `--disc-gain` | `2.5` | size of the baked discs relative to the stored ones |
| `--hole-stats` | off | count the hole pixels of every frame (also without patching) |

Tiles are cached in `cache/` and generated images in `assets_cache/`, one directory per seed and setting.

### 3. Walk

Open `http://<host>:8095/` (or `:8096`) and click the picture to capture the mouse.

| Key | Action |
|---|---|
| `W` `A` `S` `D`, `Shift` | walk, run |
| `1` | 3DGS render |
| `2` | FastFix3D |
| `3` | ground truth (ray tracer) |
| `4` | 3DGS render and FastFix3D side by side |
| `R` | next destination |
| `H` | back to the start (home) |
| `N` | navigation panel on / off |
| `Esc` | release the mouse |

`GET /routes` returns the home, the routes and their steps as JSON; `GET /blocks?x=&z=` the block outlines within 300 m of a point, for the map.

## Endless Labyrinth

See [endless-labyrinth/README.md](../endless-labyrinth/README.md).

## Sync

```bash
./sync.sh        # pulls ~/endless-labyrinth (git subtree) and ~/endless-streets (sources only) into this repository
```
