# Endless Streets: a generated street town for route training

**An endless Chinese street town you walk through in the browser, from home to everyday destinations, with a navigation panel that names a landmark at every turn. Each part of the town is turned into 3D Gaussians the first time it is needed, and every frame is repaired live by FastFix3D.**

- **Training routes:** home is a residential compound with its name plate; four routes lead to a community health centre, a market, a senior centre and a post office, each turning only at crossings with landmarks (`routes.py`).
- **Navigation panel:** instructions in Chinese that name the landmark and the street, a heading-up map, remaining distance and time, and an off-route warning (`client/walk.html`).
- **The town:** avenues, streets and lanes on an irregular grid with T-junctions; street-front buildings with podiums, set-back towers, forecourts and roof structures; residential compounds, schools, squares, market halls and parks; bus shelters, traffic lights, street-name signs, bins, shared bikes and planters (`world.py`).
- **Generated looks:** every facade (most with a shop sign), every ground, tree, lamp and piece of street furniture comes from Qwen-Image-2.1-Turbo while you walk; road markings and kerbs are drawn by code.
- **Holes closed:** at tile build time from walker-like probe views (`stiles.py`), and while walking by a background patcher that turns traced hole pixels into splats (`server.py`).
- **Same place every time:** the town and the routes are pure functions of the seed, and generated images are cached by key.
- **Views:** 3DGS render, FastFix3D, ray-traced ground truth, and render | FastFix3D side by side.

## Files

| File | Contents |
|---|---|
| `world.py` | the town: street network, blocks and their buildings, street furniture, landmarks, traffic lights and signs at crossings |
| `routes.py` | home, destinations, routes through landmark crossings, street names |
| `cityassets.py` | what each object's image is: key, prompt, size, kind |
| `assetgen.py`, `assetgen.sh` | image generator service: Qwen-Image-2.1-Turbo FP8, cut-outs, seamless textures, facade crop |
| `raytrace.py` | GPU ray tracer (Triton) with textures, alpha-tested cut-outs, sun and shadows |
| `stiles.py` | tiles: which squares are built, camera rig, splats from the rig views (baked or WorldMirror), pruning, hole patching |
| `splatrender.py` | gsplat rendering |
| `server.py`, `run.sh` | tile store, hole patcher, walker, frame pipeline, WebSocket server, `/routes`, `/blocks` |
| `client/walk.html` | the page and the navigation panel |
| `walk_test.py`, `evaltile.py` | headless walk and tile scoring |

How to run: [../docs/USAGE.md](../docs/USAGE.md).
