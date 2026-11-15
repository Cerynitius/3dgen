# 3dgen

Generated 3D worlds you can walk through in the browser. The world is a pure function of a seed; the part around the
walker is turned into 3D Gaussian splats the first time it is needed, so there is no edge and nothing is pre-built.

| Project | What you walk through |
|---|---|
| [`endless-labyrinth/`](endless-labyrinth/) | an infinite maze; tiles reconstructed on demand by a feed-forward model (WorldMirror) from ray-cast rig views, rendered client-side in WebGL2 |
| [`endless-streets/`](endless-streets/) | an endless town (street grid, shops, trees, landmarks at crossings); tiles baked or reconstructed by WorldMirror, frames streamed from the server and cleaned up live by FastFix3D; the town's looks are generated while you walk |

## Endless Streets

- **Generated looks.** Every building gets its own front-elevation photo with a Chinese shop sign, every street
  segment its own asphalt, every block its own paving, every street one tree species; statues, benches, kiosks,
  poster columns and street clocks are generated cut-outs and photos (Qwen-Image-2.1-Turbo, FP8, ~2 s per image on an
  RTX PRO 4000). Generated content covers 95% of the non-sky picture; road markings, kerbs and lamp posts are drawn by
  code.
- **Live repair.** 3DGS render → FastFix3D (TensorRT, 960×540) at 22–27 fps while walking.
- **Few holes.** Built tiles leave 0.39% of surface pixels uncovered (1.45% before interior tiles and
  distance-relative point rejection).
- **Start-up.** The 37 tiles around the start, with their ~150 generated assets, are ready in about 5 minutes.

## Layout

`endless-labyrinth/` keeps its own history (git subtree). `endless-streets/` imports the labyrinth's tile helpers from
the sibling directory. `sync.sh` pulls both working copies into this repo.
