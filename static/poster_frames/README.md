# Poster frames

Static three-frame sequences pulled from the ActionParty result videos, for use
in a printed poster. Regenerate with:

```bash
pip install imageio-ffmpeg pillow numpy
python3 tools/extract_poster_frames.py            # 512 px frames
python3 tools/extract_poster_frames.py --scale 3  # + 1536 px nearest-neighbour upscales for print
```

## How the frames were chosen

Every result video is a **5-state rollout**: one initial context frame followed
by four generated steps. Each state is held for 18 video frames while the
per-player action arrows and the player-number badges animate in, so the fully
annotated render of each state is the *last* frame of its cycle — video frames
`15, 33, 51, 69, 79`. Those become `s1 … s5`, where **`s1` is the given context
frame and `s2`–`s5` are generated**. Letterbox padding is trimmed with a single
shared crop per game, so the three frames stay pixel-registered.

Any three consecutive states form a candidate window:

| option | frames  | note                                   |
| ------ | ------- | -------------------------------------- |
| A      | s1–s3   | starts from the given context frame    |
| B      | s2–s4   | all three frames generated             |
| C      | s3–s5   | all three generated, latest drift      |

## Contents

- `<game>/s1.png … s5.png` — the five annotated states, cropped, 512 px
- `<game>/strip_sX-sY.png` — the recommended window pre-composed as a strip
- `_sheets/candidates_*.png` — every game × every window, for picking
- `_sheets/poster_set_A.png`, `_sheets/poster_set_B.png` — two whole-poster line-ups
- `manifest.json` — source video, frame indices, crop box and recommended window per game

## Games

| key | source | subjects |
| --- | ------ | -------- |
| `coins` | `static/figures/demo_ours.mp4` | 2 |
| `cooking` | `static/figures/cooking_ours.mp4` | 2 |
| `cleanup` | `static/figures/crowded_ours.mp4` | 4 |
| `mining` | `static/figures/mining_ours.mp4` | 4 |
| `predator_prey_alley_hunt` | `static/figures/predator_prey_ours.mp4` | 7 |
| `chemistry_three_cycles` | `static/figures/chemistry_ours.mp4` | 7 |
| `predator_prey_open` | `supplementary/videos/predator_prey__open_s0.mp4` (seed 0) | 7 |
| `paintball_ctf` | `supplementary/videos/paintball__capture_the_flag_s0.mp4` (seed 0) | 8 |
| `fruit_market` | `supplementary/videos/fruit_market__concentric_rivers_s0.mp4` (seed 0) | 6 |
