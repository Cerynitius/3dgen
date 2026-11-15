# Endless Streets: a generated street town for route training

**An endless Chinese street town you walk through in the browser, from home to everyday destinations, with a navigation panel that names a landmark at every turn. Each part of the town is turned into 3D Gaussians the first time it is needed, and every frame is repaired live by FastFix3D.**

- **Training routes:** home is a residential compound with its name plate; four routes lead to a community health centre, a market, a senior centre and a post office, each turning only at crossings with landmarks (`routes.py`).
- **Navigation panel:** instructions in Chinese that name the landmark and the street, a heading-up map, remaining distance and time, and an off-route warning (`client/walk.html`).
- **Generated looks:** facades with shop signs, asphalt, paving, trees, statues, benches, kiosks, poster columns and street clocks come from Qwen-Image-2.1-Turbo while you walk; road markings, kerbs and lamp posts are drawn by code.
- **Same place every time:** the town and the routes are pure functions of the seed, and generated images are cached by key.
- **Views:** 3DGS render, FastFix3D, ray-traced ground truth, and render | FastFix3D side by side.

## Files

| File | Contents |
|---|---|
| `world.py` | the town: street grid, buildings, trees, lamps, benches, landmarks at crossings |
| `routes.py` | home, destinations, routes through landmark crossings, street names |
| `cityassets.py` | what each object's image is: key, prompt, size, kind |
| `assetgen.py`, `assetgen.sh` | image generator service: Qwen-Image-2.1-Turbo FP8, cut-outs, seamless textures, facade crop |
| `raytrace.py` | GPU ray tracer (Triton) with textures, alpha-tested cut-outs, sun and shadows |
| `stiles.py` | tiles: camera rig, splats from the rig views (baked or WorldMirror), pruning |
| `splatrender.py` | gsplat rendering |
| `server.py`, `run.sh` | tile store, walker, frame pipeline, WebSocket server, `/routes` |
| `client/walk.html` | the page and the navigation panel |
| `walk_test.py`, `evaltile.py` | headless walk and tile scoring |

How to run: [../docs/USAGE.md](../docs/USAGE.md).
