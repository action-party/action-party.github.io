#!/usr/bin/env python3
"""Extract static poster frames from the ActionParty result videos.

Each result video is a 5-state rollout: an initial context frame followed by
four generated steps.  The videos hold every state for 18 video frames while
the per-player action arrows and player-number badges animate in, so the fully
annotated frame for each state is the last one of its cycle:

    video frame 15, 33, 51, 69, 79  ->  state s1 (context), s2, s3, s4, s5

This script pulls those five frames per game, trims the letterbox padding
(shared crop across all five so the sequence stays registered), and writes
    static/poster_frames/<game>/s{1..5}.png
plus a horizontal strip for the recommended 3-frame window and contact sheets
of every candidate window.

Usage:  python3 tools/extract_poster_frames.py [--scale N]

Pass --scale 3 to also write nearest-neighbour 3x upscales (1536 px) for
print: at 300 dpi a native 512 px frame is only ~1.7 in wide.
Requires: pip install imageio-ffmpeg pillow numpy
"""
import argparse
import json
import os
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "static", "poster_frames")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# video frame index of the fully annotated render of each rollout state
STATE_FRAMES = [15, 33, 51, 69, 79]

# key -> (video path, header rows to drop, caption, recommended window start)
GAMES = {
    "coins": ("static/figures/demo_ours.mp4", 0,
              "Coins (2 players)", 2),
    "cooking": ("static/figures/cooking_ours.mp4", 0,
                "Collaborative Cooking: Asymmetric (2 players)", 3),
    "cleanup": ("static/figures/crowded_ours.mp4", 0,
                "Clean Up (4 players)", 1),
    "mining": ("static/figures/mining_ours.mp4", 0,
               "Coop Mining (4 players)", 1),
    "predator_prey_alley_hunt": ("static/figures/predator_prey_ours.mp4", 32,
                                 "Predator Prey: Alley Hunt (7 players)", 1),
    "chemistry_three_cycles": ("static/figures/chemistry_ours.mp4", 32,
                               "Chemistry: Three Metabolic Cycles (7 players)", 1),
    "predator_prey_open": ("supplementary/videos/predator_prey__open_s0.mp4", 56,
                           "Predator Prey: Open (7 players)", 1),
    "paintball_ctf": ("supplementary/videos/paintball__capture_the_flag_s0.mp4", 56,
                      "Paintball: Capture the Flag (8 players)", 3),
    "fruit_market": ("supplementary/videos/fruit_market__concentric_rivers_s0.mp4", 56,
                     "Fruit Market: Concentric Rivers (6 players)", 2),
}


def extract(path, header, dest):
    """Write the five annotated state frames of `path` into `dest`."""
    os.makedirs(dest, exist_ok=True)
    select = "+".join(r"eq(n\,%d)" % i for i in STATE_FRAMES)
    # supplementary grids are 5 seeds side by side; keep the leftmost (seed 0)
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", os.path.join(REPO, path),
         "-vf", "select='%s',crop=512:512:0:%d" % (select, header),
         "-vsync", "0", os.path.join(dest, "s%d.png")],
        check=True)
    return [Image.open(os.path.join(dest, "s%d.png" % i)).convert("RGB")
            for i in range(1, 6)]


def content_box(frames, pad=0, thresh=12):
    """Bounding box of non-letterbox content, shared across all frames."""
    stack = np.stack([np.asarray(f, dtype=np.int16) for f in frames])
    lit = (stack.max(axis=3) > thresh).any(axis=0)
    rows = np.where(lit.any(axis=1))[0]
    cols = np.where(lit.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return (0, 0, frames[0].width, frames[0].height)
    h, w = lit.shape
    return (max(int(cols[0]) - pad, 0), max(int(rows[0]) - pad, 0),
            min(int(cols[-1]) + 1 + pad, w), min(int(rows[-1]) + 1 + pad, h))


def strip(frames, gutter=10, bg="white"):
    w, h = frames[0].size
    out = Image.new("RGB", (len(frames) * w + (len(frames) - 1) * gutter, h), bg)
    for i, f in enumerate(frames):
        out.paste(f, (i * (w + gutter), 0))
    return out


def sheet(entries, path, tile=185):
    """Contact sheet: one labelled row per candidate window."""
    width = 3 * tile + 150
    height = sum(3 * (tile + 6) + 22 for _ in entries)
    im = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(im)
    y = 0
    for key, caption, frames in entries:
        draw.text((4, y + 5), "%s - %s" % (key, caption), fill="black")
        y += 22
        for start in range(1, 4):
            draw.text((3 * tile + 8, y + tile // 2 - 12),
                      "option %s\ns%d-s%d" % ("ABC"[start - 1], start, start + 2),
                      fill="black")
            for j in range(3):
                f = frames[start - 1 + j]
                f = f.resize((tile, int(tile * f.height / f.width)))
                im.paste(f, (j * tile, y))
                draw.rectangle([j * tile, y, j * tile + tile - 1,
                                y + f.height - 1], outline="#bbbbbb")
            y += tile + 6
    im.save(path)


# Whole-poster line-ups to choose between: (title, [(game, window start)])
POSTER_SETS = {
    "A": ("Set A - the four site games plus Alley Hunt (7 players)",
          [("cooking", 3), ("cleanup", 1), ("mining", 1),
           ("predator_prey_alley_hunt", 1), ("coins", 2)]),
    "B": ("Set B - swaps in the light-palette and open-field rollouts",
          [("cooking", 3), ("cleanup", 1), ("predator_prey_open", 1),
           ("chemistry_three_cycles", 1), ("paintball_ctf", 3)]),
}


def poster_set(name, frames_by_game, path, tile=200):
    title, picks = POSTER_SETS[name]
    width = 3 * tile + 12 + 210
    height = len(picks) * (tile + 26) + 26
    im = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(im)
    draw.text((6, 8), title, fill="black")
    y = 26
    for game, start in picks:
        frames = frames_by_game[game][start - 1:start + 2]
        for j, f in enumerate(frames):
            f = f.resize((tile, int(tile * f.height / f.width)))
            im.paste(f, (j * (tile + 6), y))
            draw.rectangle([j * (tile + 6), y, j * (tile + 6) + tile - 1,
                            y + f.height - 1], outline="#bbbbbb")
        draw.text((3 * (tile + 6) + 8, y + tile // 2 - 16),
                  "%s\n%s\ns%d-s%d" % (game, GAMES[game][2], start, start + 2),
                  fill="black")
        y += tile + 26
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=int, default=1,
                    help="also write nearest-neighbour NxN upscales for print")
    scale = ap.parse_args().scale

    os.makedirs(os.path.join(OUT, "_sheets"), exist_ok=True)
    manifest, entries = {}, []
    for key, (path, header, caption, rec) in GAMES.items():
        dest = os.path.join(OUT, key)
        frames = extract(path, header, dest)
        box = content_box(frames)
        frames = [f.crop(box) for f in frames]
        for i, f in enumerate(frames, 1):
            f.save(os.path.join(dest, "s%d.png" % i))
            if scale > 1:
                f.resize((f.width * scale, f.height * scale),
                         Image.NEAREST).save(
                    os.path.join(dest, "s%d@%dx.png" % (i, scale)))
        strip(frames[rec - 1:rec + 2]).save(
            os.path.join(dest, "strip_s%d-s%d.png" % (rec, rec + 2)))
        manifest[key] = {
            "caption": caption,
            "source": path,
            "video_frames": STATE_FRAMES,
            "crop": list(box),
            "recommended_window": "s%d-s%d" % (rec, rec + 2),
        }
        entries.append((key, caption, frames))
        print("%-26s crop=%s  recommended s%d-s%d" % (key, box, rec, rec + 2))

    for i in range(0, len(entries), 3):
        sheet(entries[i:i + 3],
              os.path.join(OUT, "_sheets", "candidates_%d.png" % (i // 3 + 1)))
    frames_by_game = {k: f for k, _, f in entries}
    for name in POSTER_SETS:
        poster_set(name, frames_by_game,
                   os.path.join(OUT, "_sheets", "poster_set_%s.png" % name))

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
