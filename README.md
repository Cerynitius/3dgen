# Endless Labyrinth: an infinite maze reconstructed as 3D Gaussians while you walk

**A first-person labyrinth with no edge. The geometry you see is produced on demand by a feed-forward reconstruction model (WorldMirror) from a rig of ray-cast views of the cells you are approaching, and rendered as 3D Gaussian splats in the browser.**

- **Nothing pre-baked:** the maze is an infinite procedural lattice, and every tile is reconstructed the first time it is needed.
- **The same tile every time:** a tile is a pure function of `(seed, tile index, spec)`, so a tile you walk back to is the tile you left, byte for byte, and the cache can be dropped under pressure without the world changing.
- **Browser client:** a WebGL2 splat renderer in plain ES modules with no build step; collision is resolved locally from the wall bits the server sends.
- **Tile build:** about 2.5–4 s per tile on the reference machine, nearest tiles first.

```
     browser  ──  /spec   metric constants the client places tiles by
                  /maze   wall bits, so collision is resolved locally
                  /ring   which tiles surround you, nearest first
                  /tile   binary splats, 14 floats each
```

## Repository

| Path | Contents |
|---|---|
| `playground/labyrinth/` | the client: WebGL2 splat renderer, controls, collision, streaming |
| `experiments/labyrinth/endless/` | the server: maze, raycaster, camera rig, reconstruction, HTTP |

### Client

| File | Contents |
|---|---|
| `endless.js` | entry point: streaming loop, ring pump, HUD |
| `splats.js` | WebGL2 splat field: texture-backed slots, LRU eviction, depth sort worker |
| `controls.js` | first-person controller, exact exponential-approach integration |
| `occupancy.js` | odd/even occupancy grid rebuilt from the wall bits `/maze` serves |
| `netpolicy.js` | deadlines, exponential backoff, cost-keyed refusal cache |
| `geometry.js`, `collider3d.js`, `mat4.js`, `maze.js` | mesh building, collision, matrix maths, local generator |
| `endless.html` | the page |
| `live.html`, `live.js` | a mesh-rendered view of the same maze |

### Server

| File | Contents |
|---|---|
| `stream_server.py` | HTTP endpoints, tile cache, build queue, worker threads |
| `infinite_maze.py` | the infinite lattice: hashed chunk gates, occupancy rasterisation |
| `render.py` | GPU raycaster: the views the model is conditioned on, and true depth |
| `tiles.py` | tile geometry, camera rig, metric scale, cropping, pruning, wire packing |
| `recon.py` | the reconstruction model wrapper |

Design notes: `experiments/labyrinth/endless/README.md` (server design), `experiments/labyrinth/endless/CONFIG.md` (default configuration and the measurement behind each value), `playground/labyrinth/STATUS.md` (pipeline state, in Chinese).

How to run and verify: [docs/USAGE.md](docs/USAGE.md).

## Licence and attribution

The labyrinth code in this repository is its authors'. It depends on, and is designed around, Tencent's HunyuanWorld 2.0 and the WorldMirror reconstruction model, which carry their own licence; see the [upstream repository](https://github.com/Tencent-Hunyuan/HY-World-2.0). No upstream code or model weights are redistributed here.
