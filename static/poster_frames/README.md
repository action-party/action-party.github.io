# Poster frames

Static three-frame sequences pulled from the ActionParty result videos, for use
in a printed poster. Regenerate with:

```bash
pip install imageio-ffmpeg pillow numpy
python3 tools/extract_poster_frames.py
```

## How the frames were chosen

Every result video is a **5-state rollout**: one initial context frame followed
by four generated steps. Each state is held for 18 video frames while its
overlay cycles through three phases — the underlying game frame never changes
within a cycle, only the annotation on top of it, so the phases mix freely:

| phase   | video frames        | what the badge shows |
| ------- | ------------------- | -------------------- |
| `post`  | –, 23, 41, 59, 77   | the action that produced this frame, bordered **green** if the model executed it correctly and **red** if not. None for `s1`, which no action led into. |
| `clean` | 6, 25, 43, 61, 79   | nothing — orientation cones only, the plain game frame |
| `pre`   | 15, 33, 51, 69, –   | in **white**, the action about to be applied. None for `s5`, where the video ends. |

Letterbox padding is trimmed with a single shared crop per game, so the frames
stay pixel-registered.

**A window is built as `clean` → `post` → `post`.** The opening panel is the
frame handed to the model, so it carries no action badges: annotating it would
suggest those actions produced it. The two generated panels that follow are
badged with the action that produced each of them.

| option | states  | note                                     |
| ------ | ------- | ---------------------------------------- |
| A      | s1–s3   | opens on the given context frame         |
| B      | s2–s4   | opens on a generated frame               |
| C      | s3–s5   | opens on a generated frame, latest drift |

The script reports which windows contain a **red** badge — an action the model
did not execute correctly. That is usually not what you want on a poster, so
the recommended window per game avoids one wherever a red-free window exists.
Every window of `chemistry_three_cycles` contains one.

## Zoom tiers

A whole 512 px map puts each sprite at ~16 px, far too small to read in print,
so every window is also written as a zoomed crop. The crop box is picked from
what moves across the three frames and is **shared by all three**, so the
sequence stays registered; the crop is then upscaled nearest-neighbour to
~1024 px so the pixel art stays crisp.

| tier   | crop                                        | typical zoom |
| ------ | ------------------------------------------- | ------------ |
| *(whole map)* | `<game>/<phase>_sN.png`              | 1x           |
| `all`  | bounding box of every subject that moves    | 2–7x         |
| `mid`  | fixed 288 px window on the busiest cluster  | 4x           |
| `hero` | fixed 180 px window on the busiest cluster  | 6x           |

`all` keeps the whole map legible but barely zooms on the wide maps, where the
subjects are spread corner to corner. `hero` trades that context for sprites
and action badges big enough to read across a room — it is the tier used for
the whole-poster line-ups.

## Contents

- `<game>/{clean,pre,post}_sN.png` — every state in every phase, whole map
- `<game>/strip_sX-sY.png` — the recommended window as a whole-map strip
- `<game>/{all,mid,hero}_sX-sY/` — the recommended window at each zoom tier,
  as three frames plus a pre-composed `strip.png`
- `_sheets/candidates_*.png` — every game × every window, whole map
- `_sheets/{all,mid,hero}_candidates_*.png` — the same at each zoom tier
- `_sheets/poster_set_A.png`, `_sheets/poster_set_B.png` — two line-ups
- `manifest.json` — source video, phase frame indices, crop boxes, upscale
  factors, and the green/red badge pixel counts behind the red-badge warning

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
