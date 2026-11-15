# 3dgen

Walkable 3D street worlds for route training of older adults with mild cognitive impairment (MCI) or early
Alzheimer's disease, built on the FastFix3D pipeline (real-time repair of 3D Gaussian Splatting walkthroughs).

Getting lost is common and often early in Alzheimer's disease (61 of 112 patients living at home had trouble finding
their way; for 28 it was one of the first signs, Pai & Jacobs 2004). Patients still recognize landmarks but cannot
tie them to places and turns (deIpolyi et al. 2007), and practice in a virtual town carries over to the real one,
better when the copy is detailed (Wallet et al. 2011). The worlds here are made to serve that kind of practice:

| What route training needs | How the system provides it |
|---|---|
| **Landmarks at decision points.** The brain stores what stands at a crossing without effort (Janzen & van Turennout 2004); salient cues at decision points helped people with AD and MCI find their way (Davis et al. 2017). | About half of the crossings carry a landmark: a statue on a roundabout island, a kiosk, an advertising column, a street clock or a big corner tree. Each street has one tree species, and each shop its own Chinese sign, so streets can be told apart. |
| **The same place every time.** Route practice is repeated study of the same scenes, and what fails in AD is linking a scene to its place. | The town is a pure function of a seed and the block index. Each generated photo is made once and cached by its key, so a street you walk back to looks exactly as you left it, and a route can be repeated day after day. |
| **No invented details.** People with memory problems learn their own errors (Baddeley & Wilson 1994); a made-up shop sign at a crossing is a wrong lesson. | The ray-traced town is the ground truth for every frame. FastFix3D keeps the 3D render and adds only a correction, and the ground truth view is available live to check it. |
| **Smooth, fast pictures.** Delay and flicker cause cybersickness (LaViola 2000). | 3DGS render → FastFix3D (TensorRT, 960×540) at 22–27 fps on one GPU while walking. |
| **No expert work per place.** Hand-built virtual places are why training has not reached people's own routes. | Everything is generated: layout from hashes, appearance from an image model while you walk. A new start point is ready in about 5 minutes. |

## Projects

| | |
|---|---|
| [`endless-streets/`](endless-streets/) | An endless Chinese street town. 95% of the non-sky picture is generated live: each building's facade and shop sign, each street's asphalt, each block's paving, trees, statues, benches, kiosks, poster columns (community notices such as university-for-the-elderly enrolment) and street clocks, made with Qwen-Image-2.1-Turbo (FP8, ~2 s per image). Road markings, kerbs and lamp posts are drawn by code. Tiles are baked or reconstructed by WorldMirror and streamed to the browser through FastFix3D. Built tiles leave 0.39% of surface pixels uncovered. |
| [`endless-labyrinth/`](endless-labyrinth/) | The predecessor: an infinite maze whose tiles are reconstructed on demand by WorldMirror from ray-cast rig views and rendered in the browser with WebGL2. It supplies the tile machinery the street town builds on. |

## Status and limits

- **Not the person's own route.** The background research asks for a copy of the person's own route (familiar
  places protect against disorientation, Lopez et al. 2018). The generated town gives unlimited, repeatable practice
  routes with known ground truth; copies of real routes come from walking videos through FastFix3D (the ISEF paper).
- **Not headset rate yet.** 22–27 fps is enough on a screen; a headset needs a much higher rate.
- **No study with older adults yet.** Whether practice in these worlds helps people with MCI on real streets has to
  be shown by a study with them.
- Generated imperfections: seam ghosting on regular paving, flat bench cut-outs, procedural kerbs and markings.

## Layout

`endless-labyrinth/` keeps its own history (git subtree). `endless-streets/` imports the labyrinth's tile helpers from
the sibling directory. `sync.sh` pulls both working copies into this repo. Usage: `endless-streets/README.md`.

## References

- Baddeley, A., & Wilson, B. A. (1994). When implicit learning fails: Amnesia and the problem of error elimination. *Neuropsychologia*.
- Davis, R., Ohman, J. M., & Weisbeck, C. (2017). Salient cues and wayfinding in Alzheimer's disease within a virtual senior residence. *Environment and Behavior*.
- deIpolyi, A. R., Rankin, K. P., Mucke, L., Miller, B. L., & Gorno-Tempini, M. L. (2007). Spatial cognition and the human navigation network in AD and MCI. *Neurology*.
- Janzen, G., & van Turennout, M. (2004). Selective neural representation of objects relevant for navigation. *Nature Neuroscience*.
- LaViola, J. J. (2000). A discussion of cybersickness in virtual environments. *ACM SIGCHI Bulletin*.
- Lopez, A., Caffò, A. O., & Bosco, A. (2018). Topographical disorientation in aging. Familiarity with the environment does matter. *Neurological Sciences*.
- Pai, M.-C., & Jacobs, W. J. (2004). Topographical disorientation in community-residing patients with Alzheimer's disease. *International Journal of Geriatric Psychiatry*.
- Wallet, G., Sauzéon, H., Pala, P. A., et al. (2011). Virtual/real transfer of spatial knowledge: Benefit from visual fidelity provided in a virtual environment. *Cyberpsychology, Behavior, and Social Networking*.
