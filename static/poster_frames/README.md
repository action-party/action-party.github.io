# Poster frames

Static three-frame sequences pulled from the ActionParty result videos, for use
in a printed poster. Regenerate with:

```bash
pip install imageio-ffmpeg pillow numpy
python3 tools/extract_poster_frames.py
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

## Zoom tiers

A whole 512 px map puts each sprite at ~16 px, far too small to read in print,
so every window is also written as a zoomed crop. The crop box is picked from
what moves across the three frames and is **shared by all three**, so the
sequence stays registered; the crop is then upscaled nearest-neighbour to
~1024 px so the pixel art stays crisp.

| tier   | crop                                        | typical zoom |
| ------ | ------------------------------------------- | ------------ |
| *(whole map)* | `<game>/s1.png … s5.png`             | 1x           |
| `all`  | bounding box of every subject that moves    | 2–7x         |
| `mid`  | fixed 288 px window on the busiest cluster  | 4x           |
| `hero` | fixed 180 px window on the busiest cluster  | 6x           |

`all` keeps the whole map legible but barely zooms on the wide maps, where the
subjects are spread corner to corner. `hero` trades that context for sprites
and action badges big enough to read across a room — it is the tier used for
the whole-poster line-ups.

## Contents

- `<game>/s1.png … s5.png` — the five annotated states, whole map, cropped
- `<game>/strip_sX-sY.png` — the recommended window as a whole-map strip
- `<game>/{all,mid,hero}_sX-sY/` — the recommended window at each zoom tier,
  as three frames plus a pre-composed `strip.png`
- `_sheets/candidates_*.png` — every game × every window, whole map
- `_sheets/{all,mid,hero}_candidates_*.png` — the same at each zoom tier
- `_sheets/poster_set_A.png`, `_sheets/poster_set_B.png` — two line-ups
- `manifest.json` — source video, frame indices, crop boxes and upscale factors

To write out a different window for a game, change its last field in `GAMES` in
`tools/extract_poster_frames.py` and re-run; the contact sheets always cover all
three. `--zoom-width` sets the target width of a zoomed frame, `--scale N` also
writes NxN upscales of the whole map, and `--poster-tier` picks the tier the
line-ups are built from.

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
