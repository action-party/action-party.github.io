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

Whole-map frames render the subjects far too small for a printed poster, so
each window is also written as a zoomed crop: the crop box is the bounding box
of everything that moves across the three frames (subjects, their action arrows
and their badges), shared by all three so the sequence stays registered, then
upscaled with nearest-neighbour so the pixel art stays crisp.

Usage:  python3 tools/extract_poster_frames.py [--scale N] [--zoom-width PX]

Pass --scale 3 to also write nearest-neighbour 3x upscales (1536 px) of the
whole map: at 300 dpi a native 512 px frame is only ~1.7 in wide.
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
                "Collaborative Cooking: Asymmetric (2 players)", 2),
    "cleanup": ("static/figures/crowded_ours.mp4", 0,
                "Clean Up (4 players)", 1),
    "mining": ("static/figures/mining_ours.mp4", 0,
               "Coop Mining (4 players)", 2),
    "predator_prey_alley_hunt": ("static/figures/predator_prey_ours.mp4", 32,
                                 "Predator Prey: Alley Hunt (7 players)", 2),
    "chemistry_three_cycles": ("static/figures/chemistry_ours.mp4", 32,
                               "Chemistry: Three Metabolic Cycles (7 players)", 3),
    "predator_prey_open": ("supplementary/videos/predator_prey__open_s0.mp4", 56,
                           "Predator Prey: Open (7 players)", 3),
    "paintball_ctf": ("supplementary/videos/paintball__capture_the_flag_s0.mp4", 56,
                      "Paintball: Capture the Flag (8 players)", 3),
    "fruit_market": ("supplementary/videos/fruit_market__concentric_rivers_s0.mp4", 56,
                     "Fruit Market: Concentric Rivers (6 players)", 3),
}


# Each rollout state is held for 18 video frames while its overlay cycles
# through three phases.  Same underlying game frame throughout - only the
# annotation changes - so the phases can be mixed freely across a sequence.
#
#   post   badge shows the action that produced this frame, bordered green or
#          red for whether the model executed it correctly (none for s1, which
#          no action led into)
#   clean  orientation cones only, no action badges - the plain game frame
#   pre    badge shows, in white, the action about to be applied to this frame
#          (none for s5, where the video ends)
PHASES = {
    "post": [None, 23, 41, 59, 77],
    "clean": [6, 25, 43, 61, 79],
    "pre": [15, 33, 51, 69, None],
}

# Frame used to place the crop boxes: annotated, so cones and badges stay in
# shot.  Pre-action everywhere it exists, the verdict frame for the last state.
LAYOUT_PHASE = ["pre", "pre", "pre", "pre", "post"]


def window_frames(states, start):
    """The three panels of a window: a clean frame, then two annotated ones.

    The opening panel is the frame handed to the model, so it carries no action
    badges - annotating it would suggest those actions produced it.  The two
    generated panels that follow are badged with the action that produced them.
    """
    return [states["clean"][start - 1],
            states["post"][start],
            states["post"][start + 1]]


def extract(path, header, dest):
    """Write every annotation phase of every state of `path` into `dest`."""
    os.makedirs(dest, exist_ok=True)
    wanted = sorted({i for idx in PHASES.values() for i in idx if i is not None})
    select = "+".join(r"eq(n\,%d)" % i for i in wanted)
    # supplementary grids are 5 seeds side by side; keep the leftmost (seed 0)
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", os.path.join(REPO, path),
         "-vf", "select='%s',crop=512:512:0:%d" % (select, header),
         "-vsync", "0", os.path.join(dest, "raw%d.png")],
        check=True)
    raw = {n: Image.open(os.path.join(dest, "raw%d.png" % (i + 1))).convert("RGB")
           for i, n in enumerate(wanted)}
    return {phase: [None if i is None else raw[i] for i in idx]
            for phase, idx in PHASES.items()}


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


def motion_mask(frames, thresh=18):
    """Pixels that change anywhere across the window - subjects and overlays."""
    arr = [np.asarray(f, dtype=np.int16) for f in frames]
    mask = np.zeros(arr[0].shape[:2], bool)
    for a, b in zip(arr, arr[1:]):
        mask |= np.abs(a - b).mean(axis=2) > thresh
    return mask


def denoise(mask, block=8, keep=0.15):
    """Drop isolated codec speckle by requiring a block to be mostly lit."""
    h, w = mask.shape
    bh, bw = h // block, w // block
    coarse = mask[:bh * block, :bw * block].reshape(
        bh, block, bw, block).mean(axis=(1, 3)) >= keep
    out = np.zeros_like(mask)
    out[:bh * block, :bw * block] = np.kron(coarse, np.ones((block, block), bool))
    return out


def action_box(frames, margin=26, min_side=150, square=True):
    """Crop box around everything that moves, shared across the window."""
    w, h = frames[0].size
    mask = denoise(motion_mask(frames))
    if not mask.any():
        return (0, 0, w, h)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    x0, x1 = int(cols[0]) - margin, int(cols[-1]) + 1 + margin
    y0, y1 = int(rows[0]) - margin, int(rows[-1]) + 1 + margin
    if square:
        side = max(x1 - x0, y1 - y0, min_side)
        side = min(side, min(w, h))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        x1, y1 = x0 + side, y0 + side
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    if square:  # re-anchor if clamping pushed a side out of the frame
        side = min(x1 - x0, y1 - y0)
        x0, y0 = min(x0, w - side), min(y0, h - side)
        x1, y1 = x0 + side, y0 + side
    return (x0, y0, x1, y1)


def components(mask):
    """Connected components (8-connected) of a boolean mask, as point arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack, pts = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < h and 0 <= nx < w
                                and mask[ny, nx] and not seen[ny, nx]):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            out.append(np.array(pts))
    return out


def badge_discs(clean, post, thresh=40, min_area=150):
    """Locate and read every action badge on an annotated frame.

    The clean and annotated renders of a state share the same underlying game
    frame, so differencing them isolates the badges exactly - no disc-finding
    heuristics needed.  Within a disc, position separates the two uses of red:
    a red *ring* is the verdict that the model got the action wrong, while a red
    *centre* is the star glyph of the Interact action itself.
    """
    a = np.asarray(post, dtype=np.int16)
    mask = np.abs(a - np.asarray(clean, dtype=np.int16)).max(axis=2) > thresh
    discs = []
    for pts in components(mask):
        if len(pts) < min_area:
            continue
        cy, cx = pts.mean(0)
        dist = np.hypot(pts[:, 0] - cy, pts[:, 1] - cx)
        r = dist.max()
        if not 8 <= r <= 40:
            continue
        px = a[pts[:, 0], pts[:, 1]].astype(np.int32)

        def frac(sel, test):
            q = px[sel]
            return float(test(q).mean()) if len(q) else 0.0

        is_red = lambda q: (q[:, 0] > q[:, 1] + 55) & (q[:, 0] > q[:, 2] + 55)
        is_green = lambda q: (q[:, 1] > q[:, 0] + 40) & (q[:, 1] > q[:, 2] + 40)
        ring, core = dist > 0.72 * r, dist < 0.55 * r
        discs.append({
            "cy": float(cy), "cx": float(cx), "r": float(r),
            "interact": frac(core, is_red) > 0.4,
            "wrong": frac(ring, is_red) > 0.5,
            "correct": frac(ring, is_green) > 0.35,
        })
    return discs


def crowd_box(discs, size, sides=range(180, 401, 20), step=8, margin=6):
    """Smallest crop holding the most badges - and, at a tie, the most Interacts.

    The fixed-side `hero` crop shows sprites large but on a wide map catches
    only two or three of the players.  This grows the window just far enough to
    gather the rest of the party back into shot.
    """
    w, h = size
    best = None
    for side in sides:
        side = min(side, w, h)
        for y0 in range(0, h - side + 1, step):
            for x0 in range(0, w - side + 1, step):
                inside = [d for d in discs
                          if x0 + margin <= d["cx"] - d["r"]
                          and d["cx"] + d["r"] <= x0 + side - margin
                          and y0 + margin <= d["cy"] - d["r"]
                          and d["cy"] + d["r"] <= y0 + side - margin]
                score = (len(inside), sum(d["interact"] for d in inside), -side)
                if best is None or score > best[0]:
                    best = (score, (x0, y0, x0 + side, y0 + side))
        if best and best[0][0] == len(discs):
            break                      # everyone is in shot; no need to grow
    return best[1] if best else (0, 0, w, h)


def save_png(im, path, colors=256):
    """Save pixel art as an indexed PNG - a fraction of the size, no visible loss."""
    if im.mode == "RGB" and len(im.getcolors(1 << 24) or [1] * (colors + 1)) > colors:
        im = im.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE)
    im.save(path, optimize=True)


def cluster_box(frames, side, thresh=18):
    """Tightest fixed-size window over the busiest cluster of subjects.

    `action_box` keeps every subject in shot, which on the wide maps barely
    zooms at all.  This slides a fixed `side` px window and keeps the position
    covering the most motion, trading full-map context for sprites large
    enough to read in print.
    """
    w, h = frames[0].size
    side = min(side, w, h)
    mask = denoise(motion_mask(frames, thresh)).astype(np.int32)
    integral = np.pad(mask, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    window = (integral[side:, side:] - integral[:-side, side:]
              - integral[side:, :-side] + integral[:-side, :-side])
    y0, x0 = np.unravel_index(int(window.argmax()), window.shape)
    return (int(x0), int(y0), int(x0) + side, int(y0) + side)


# Zoom tiers, loosest first.  `None` fits every subject that moves in the
# window; `"crowd"` is the smallest crop still holding the most action badges;
# a number is the side in px of a fixed crop over the busiest cluster.
TIERS = {"all": None, "crowd": "crowd", "mid": 288, "hero": 180}


def zoom_window(frames, side, target=1024, apply_to=None, discs=()):
    """Crop the window to the action and upscale it to `target` px wide.

    `frames` is what the crop box is measured on; `apply_to` is what gets
    cropped, when the panels being shown differ from the ones being measured.
    """
    if side is None:
        box = action_box(frames)
    elif side == "crowd":
        box = crowd_box(discs, frames[0].size)
    else:
        box = cluster_box(frames, side)
    crops = [f.crop(box) for f in (apply_to or frames)]
    factor = max(1, int(round(target / crops[0].width)))
    if factor > 1:
        crops = [c.resize((c.width * factor, c.height * factor), Image.NEAREST)
                 for c in crops]
    return box, factor, crops


def strip(frames, gutter=10, bg="white"):
    w, h = frames[0].size
    out = Image.new("RGB", (len(frames) * w + (len(frames) - 1) * gutter, h), bg)
    for i, f in enumerate(frames):
        out.paste(f, (i * (w + gutter), 0))
    return out


def sheet(entries, path, tile=185):
    """Contact sheet: one labelled row per candidate window.

    Each entry carries the nine frames of the three candidate windows.
    """
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
                f = frames[(start - 1) * 3 + j]
                f = f.resize((tile, int(tile * f.height / f.width)))
                im.paste(f, (j * tile, y))
                draw.rectangle([j * tile, y, j * tile + tile - 1,
                                y + f.height - 1], outline="#bbbbbb")
            y += tile + 6
    save_png(im, path)


# Whole-poster line-ups to choose between: (title, [(game, window start)])
POSTER_SETS = {
    "A": ("Set A - the four site games plus Alley Hunt (7 players)",
          [("cooking", 2), ("cleanup", 1), ("mining", 2),
           ("predator_prey_alley_hunt", 2), ("coins", 2)]),
    "B": ("Set B - swaps in the light-palette and open-field rollouts",
          [("cooking", 2), ("cleanup", 1), ("predator_prey_open", 3),
           ("chemistry_three_cycles", 3), ("paintball_ctf", 3)]),
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
        frames = frames_by_game[game][(start - 1) * 3:(start - 1) * 3 + 3]
        for j, f in enumerate(frames):
            f = f.resize((tile, int(tile * f.height / f.width)))
            im.paste(f, (j * (tile + 6), y))
            draw.rectangle([j * (tile + 6), y, j * (tile + 6) + tile - 1,
                            y + f.height - 1], outline="#bbbbbb")
        draw.text((3 * (tile + 6) + 8, y + tile // 2 - 16),
                  "%s\n%s\ns%d-s%d" % (game, GAMES[game][2], start, start + 2),
                  fill="black")
        y += tile + 26
    save_png(im, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=int, default=1,
                    help="also write nearest-neighbour NxN upscales of the whole map")
    ap.add_argument("--zoom-width", type=int, default=1024,
                    help="target width in px for every zoomed frame")
    ap.add_argument("--poster-tier", default="hero", choices=list(TIERS),
                    help="zoom tier used for the whole-poster line-ups")
    args = ap.parse_args()

    os.makedirs(os.path.join(OUT, "_sheets"), exist_ok=True)
    manifest = {}
    wide = {}                        # game -> three whole-map panels per window
    tiered = {t: {} for t in TIERS}  # tier -> game -> nine zoomed panels
    for key, (path, header, caption, rec) in GAMES.items():
        dest = os.path.join(OUT, key)
        states = extract(path, header, dest)
        box = content_box([f for f in states["clean"] if f is not None])
        states = {phase: [None if f is None else f.crop(box) for f in frames]
                  for phase, frames in states.items()}
        for phase, frames in states.items():
            for i, f in enumerate(frames, 1):
                if f is None:
                    continue
                save_png(f, os.path.join(dest, "%s_s%d.png" % (phase, i)))
                if args.scale > 1:
                    save_png(f.resize((f.width * args.scale, f.height * args.scale),
                                      Image.NEAREST),
                             os.path.join(dest, "%s_s%d@%dx.png"
                                          % (phase, i, args.scale)))
        for stale in os.listdir(dest):       # ffmpeg scratch
            if stale.startswith("raw"):
                os.remove(os.path.join(dest, stale))

        layout = [states[p][i] for i, p in enumerate(LAYOUT_PHASE)]
        discs = [None] + [badge_discs(states["clean"][i], states["post"][i])
                          for i in range(1, 5)]
        zooms = {}
        for start in range(1, 4):
            panels = window_frames(states, start)
            wide.setdefault(key, []).extend(panels)
            if start == rec:
                save_png(strip(panels),
                         os.path.join(dest, "strip_s%d-s%d.png" % (start, start + 2)))
            for tier, side in TIERS.items():
                # place the crop on the annotated frames so no badge is clipped,
                # then apply it to the panels actually being shown
                zbox, factor, crops = zoom_window(
                    layout[start - 1:start + 2], side, target=args.zoom_width,
                    apply_to=panels,
                    discs=discs[start] + discs[start + 1])
                tiered[tier].setdefault(key, []).extend(crops)
                if start != rec:     # only the recommended window is written out
                    continue
                tag = "%s_s%d-s%d" % (tier, start, start + 2)
                zdir = os.path.join(dest, tag)
                os.makedirs(zdir, exist_ok=True)
                for i, c in enumerate(crops, start):
                    save_png(c, os.path.join(zdir, "s%d.png" % i))
                save_png(strip(crops, gutter=16), os.path.join(zdir, "strip.png"))
                zooms[tier] = {"crop": list(zbox), "upscale": factor,
                               "px": crops[0].width}

        manifest[key] = {
            "caption": caption,
            "source": path,
            "phase_frames": PHASES,
            "crop": list(box),
            "recommended_window": "s%d-s%d" % (rec, rec + 2),
            "panels": "clean s%d, post s%d, post s%d" % (rec, rec + 1, rec + 2),
            "badges": {"s%d" % (i + 1): {
                "players": len(d),
                "interact": sum(x["interact"] for x in d),
                "wrong": sum(x["wrong"] for x in d),
            } for i, d in enumerate(discs) if d is not None},
            "zoom": zooms,
        }
        def tally(w, field):
            return sum(x[field] for i in (w, w + 1) for x in discs[i])

        print("%-26s s%d-s%d  %s" % (
            key, rec, rec + 2,
            "  ".join("%s=%dpx@%dx" % (t, z["crop"][2] - z["crop"][0],
                                       z["upscale"]) for t, z in zooms.items())))
        for w in range(1, 4):
            print("%-26s   s%d-s%d  badges=%-2d interact=%-2d wrong=%d%s" % (
                "", w, w + 2,
                sum(len(discs[i]) for i in (w, w + 1)),
                tally(w, "interact"), tally(w, "wrong"),
                "  <- recommended" if w == rec else ""))

    keys = list(GAMES)
    for i in range(0, len(keys), 3):
        group = keys[i:i + 3]
        sheet([(k, GAMES[k][2], wide[k]) for k in group],
              os.path.join(OUT, "_sheets", "candidates_%d.png" % (i // 3 + 1)))
        for tier in TIERS:
            sheet([(k, GAMES[k][2], tiered[tier][k]) for k in group],
                  os.path.join(OUT, "_sheets",
                               "%s_candidates_%d.png" % (tier, i // 3 + 1)))

    for name in POSTER_SETS:
        poster_set(name, tiered[args.poster_tier],
                   os.path.join(OUT, "_sheets", "poster_set_%s.png" % name))

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
