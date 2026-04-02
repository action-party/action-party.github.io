#!/usr/bin/env python3
"""Generate supplementary materials HTML.

Self-contained script — reads game list from videos/ and game prompts
from prompts.json.  No external dependencies beyond the Python stdlib.

Usage:
    python generate.py          # writes index.html next to this script
"""
import glob
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE, "videos")
PROMPT_FILE = os.path.join(BASE, "prompts.json")


def load_prompts():
    with open(PROMPT_FILE) as f:
        return json.load(f)


def get_games():
    files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*_s0.mp4")))
    return [os.path.basename(f).replace("_s0.mp4", "") for f in files]


def get_long_horizon_videos():
    lh_dir = os.path.join(VIDEO_DIR, "long_horizon")
    if not os.path.isdir(lh_dir):
        return []
    files = sorted(glob.glob(os.path.join(lh_dir, "*.mp4")))
    return [os.path.basename(f) for f in files]


def fmt(game_id):
    return game_id.replace("__", ": ").replace("_", " ").title()


def generate_html():
    games = get_games()
    prompts = load_prompts()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Supplementary Materials</title>
<style>
  :root {{
    --bg: #f8f9fa;
    --card-bg: #fff;
    --border: #dee2e6;
    --text: #1a1a1a;
    --muted: #6c757d;
    --accent: #2563eb;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2.5rem 1rem;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; font-weight: 700; text-align: center; margin-bottom: 0.4rem; }}
  .subtitle {{
    text-align: center; color: var(--muted); font-size: 0.9rem;
    margin-bottom: 1.5rem; max-width: 800px; margin-left: auto; margin-right: auto;
  }}
  .legend {{
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;
    font-size: 0.85rem; color: var(--muted); line-height: 1.7;
  }}
  .legend strong {{ color: var(--text); }}
  .legend-row {{ display: flex; gap: 2rem; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.4rem; }}
  .dot {{
    display: inline-block; width: 12px; height: 12px;
    border-radius: 50%; border: 2px solid; flex-shrink: 0;
  }}
  .dot-white {{ border-color: #999; background: #fff; }}
  .dot-green {{ border-color: #1ea832; background: #d4edda; }}
  .dot-red   {{ border-color: #dc3545; background: #f8d7da; }}
  .toc {{
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 2rem;
  }}
  .toc h3 {{
    font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--muted); margin-bottom: 0.5rem;
  }}
  .toc-grid {{ display: flex; flex-wrap: wrap; gap: 0.25rem 1.25rem; }}
  .toc-grid a {{ font-size: 0.82rem; color: var(--accent); text-decoration: none; }}
  .toc-grid a:hover {{ text-decoration: underline; }}
  .game-section {{
    margin-bottom: 2rem; background: var(--card-bg);
    border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
  }}
  .game-header {{
    background: #f1f3f5; padding: 0.6rem 1.25rem;
    border-bottom: 1px solid var(--border); cursor: pointer;
    user-select: none; display: flex; align-items: center;
  }}
  .game-header:hover {{ background: #e9ecef; }}
  .game-header h2 {{ font-size: 1rem; font-weight: 600; flex: 1; }}
  .game-header .toggle {{
    font-size: 0.75rem; color: var(--muted); transition: transform 0.2s;
  }}
  .game-header.collapsed .toggle {{ transform: rotate(-90deg); }}
  .game-body {{ padding: 1rem 1.25rem; }}
  .game-body.hidden {{ display: none; }}
  .game-prompt {{
    font-size: 0.82rem; color: var(--muted); font-style: italic;
    margin-bottom: 1rem; line-height: 1.5;
  }}
  .game-prompt strong {{
    color: var(--text); font-style: normal; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.03em;
  }}
  .video-block {{ margin-bottom: 1.25rem; }}
  .video-block:last-child {{ margin-bottom: 0; }}
  .video-block h3 {{
    font-size: 0.85rem; font-weight: 600; color: var(--muted);
    margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.03em;
  }}
  .video-block video {{
    width: 100%; max-width: 1100px; border-radius: 4px;
    background: #000; display: block;
  }}
  .missing {{
    padding: 2rem; text-align: center; color: #bbb; font-size: 0.85rem;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>Supplementary Materials</h1>
  <p class="subtitle">Composite action-annotated videos across all {len(games)} Melting Pot environments.
  Each video shows Ground Truth, Ours, Text-Action, Pretrained AR, and Zero-shot I2V side by side.</p>

  <div class="legend">
    <p style="margin-bottom: 0.6rem; color: var(--text); font-size: 0.85rem; line-height: 1.6;">
      Each video annotates the <strong>prescribed action</strong> for every player at each timestep.
      Before the action is applied, icons have a <strong style="color:#999">white</strong> border;
      after the action, the border turns
      <strong style="color:#1ea832">green</strong> (correct) or
      <strong style="color:#dc3545">red</strong> (incorrect),
      indicating whether the generated frame matches the expected effect.
      Colored circle markers show the model&rsquo;s <strong>predicted coordinates</strong> for each player.
    </p>
    <div style="margin-bottom: 0.5rem;">
      <strong style="font-size: 0.8rem;">Action Icons</strong> <span style="font-size: 0.75rem; color: var(--muted);">(relative to player orientation)</span>
    </div>
    <div class="legend-row" style="margin-bottom: 0.5rem;">
      <div class="legend-item">
        <svg width="28" height="28" viewBox="0 0 28 28"><circle cx="14" cy="14" r="12" fill="#fff" stroke="#999" stroke-width="2"/><text x="14" y="15" text-anchor="middle" dominant-baseline="central" font-family="DejaVu Sans,sans-serif" font-weight="bold" font-size="14" fill="#222">F</text></svg>
        Forward
      </div>
      <div class="legend-item">
        <svg width="28" height="28" viewBox="0 0 28 28"><circle cx="14" cy="14" r="12" fill="#fff" stroke="#999" stroke-width="2"/><text x="14" y="15" text-anchor="middle" dominant-baseline="central" font-family="DejaVu Sans,sans-serif" font-weight="bold" font-size="14" fill="#222">B</text></svg>
        Backward
      </div>
      <div class="legend-item">
        <svg width="28" height="28" viewBox="0 0 28 28"><circle cx="14" cy="14" r="12" fill="#fff" stroke="#999" stroke-width="2"/><text x="14" y="15" text-anchor="middle" dominant-baseline="central" font-family="DejaVu Sans,sans-serif" font-weight="bold" font-size="14" fill="#222">L</text></svg>
        Strafe Left
      </div>
      <div class="legend-item">
        <svg width="28" height="28" viewBox="0 0 28 28"><circle cx="14" cy="14" r="12" fill="#fff" stroke="#999" stroke-width="2"/><text x="14" y="15" text-anchor="middle" dominant-baseline="central" font-family="DejaVu Sans,sans-serif" font-weight="bold" font-size="14" fill="#222">R</text></svg>
        Strafe Right
      </div>
      <div class="legend-item">
        <svg width="28" height="28" viewBox="0 0 28 28"><circle cx="14" cy="14" r="12" fill="#fff" stroke="#999" stroke-width="2"/><text x="14" y="15" text-anchor="middle" dominant-baseline="central" font-family="DejaVu Sans,sans-serif" font-weight="bold" font-size="11" fill="#222">TL</text></svg>
        Turn Left
      </div>
      <div class="legend-item">
        <svg width="28" height="28" viewBox="0 0 28 28"><circle cx="14" cy="14" r="12" fill="#fff" stroke="#999" stroke-width="2"/><text x="14" y="15" text-anchor="middle" dominant-baseline="central" font-family="DejaVu Sans,sans-serif" font-weight="bold" font-size="11" fill="#222">TR</text></svg>
        Turn Right
      </div>
      <div class="legend-item">
        <svg width="28" height="28" viewBox="0 0 28 28"><circle cx="14" cy="14" r="12" fill="#fff" stroke="#999" stroke-width="2"/><polygon points="14,4 16.5,11 24,11 18,15.5 20,23 14,18.5 8,23 10,15.5 4,11 11.5,11" fill="#c8221e"/></svg>
        Interact
      </div>
      <div class="legend-item">
        <svg width="28" height="28" viewBox="0 0 28 28"><circle cx="14" cy="14" r="12" fill="#fff" stroke="#999" stroke-width="2"/><circle cx="14" cy="14" r="3.5" fill="#aaa"/></svg>
        Noop
      </div>
    </div>
    <div style="margin-bottom: 0.5rem;">
      <strong style="font-size: 0.8rem;">Icon Border Color</strong>
    </div>
    <div class="legend-row" style="margin-bottom: 0.5rem;">
      <div class="legend-item"><span class="dot dot-white"></span> Pre-action (about to apply)</div>
      <div class="legend-item"><span class="dot dot-green"></span> Correct effect</div>
      <div class="legend-item"><span class="dot dot-red"></span> Incorrect effect</div>
    </div>
    <div style="margin-bottom: 0.5rem;">
      <strong style="font-size: 0.8rem;">Baseline Comparison Layout</strong>
    </div>
    <div style="font-size: 0.82rem; color: var(--muted);">
      GT &nbsp;|&nbsp; Ours &nbsp;|&nbsp; Text-Action &nbsp;|&nbsp; Pretrained AR &nbsp;|&nbsp; Zero-shot I2V.
      &nbsp; Zero-shot I2V shows only the single next-frame prediction, then black for remaining steps.
    </div>
  </div>

  <div class="toc">
    <h3>Sections</h3>
    <div class="toc-grid">
      <a href="#long-horizon"><strong>Long Horizon Generation</strong></a>
      <a href="#variable-players"><strong>Variable Player Count</strong></a>
      <a href="#ablation"><strong>Ablation Study (Coins)</strong></a>
      <a href="#qualitative"><strong>Qualitative Results (All Games)</strong></a>
"""
    for game in games:
        html += f'      <a href="#{game}">{fmt(game)}</a>\n'

    html += """    </div>
  </div>
"""

    # --- Long Horizon section ---
    lh_grid = "videos/long_horizon/long_horizon_coins_grid.mp4"
    lh_grid_exists = os.path.exists(os.path.join(BASE, lh_grid))
    html += """
  <div class="game-section" id="long-horizon">
    <div class="game-header" onclick="toggleGame(this)">
      <h2>Long Horizon Generation &mdash; Coins (20 Actions, Sliding Window)</h2>
      <span class="toggle">&#9660;</span>
    </div>
    <div class="game-body">
      <p class="game-prompt" style="font-style: normal; color: var(--text); font-size: 0.85rem;">
        We demonstrate that our model, trained on 4-action sequences (5 latent frames),
        can generate coherent <strong>20-action rollouts</strong> via sliding-window autoregressive inference.
        The grid shows Ground Truth (top row) vs. Ours (bottom row) across 5 seeds,
        with action icons annotated at player positions.
      </p>
      <div class="video-block">
        <h3>GT vs. Ours &mdash; 5 Seeds &times; 20 Actions</h3>
"""
    if lh_grid_exists:
        html += f'        <video controls muted preload="none"><source src="{lh_grid}" type="video/mp4"></video>\n'
    else:
        html += '        <div class="missing">Long horizon grid video not available</div>\n'
    html += """      </div>
    </div>
  </div>
"""
    # --- Variable Player Count section ---
    vp_path = "videos/variable_players/variable_players_grid.mp4"
    vp_exists = os.path.exists(os.path.join(BASE, vp_path))
    html += """
  <div class="game-section" id="variable-players">
    <div class="game-header" onclick="toggleGame(this)">
      <h2>Variable Player Count &mdash; Coins (1&ndash;8 Players)</h2>
      <span class="toggle">&#9660;</span>
    </div>
    <div class="game-body">
      <p class="game-prompt" style="font-style: normal; color: var(--text); font-size: 0.85rem;">
        We test our model&rsquo;s ability to generalize to unseen player counts.
        The model was trained on games with 2&ndash;7 players (padded to 8).
        Here we generate Coins rollouts with 1 to 8 active players using the same checkpoint,
        demonstrating compositional generalization across player counts.
        Each row shows a different player count; columns are independent seeds.
      </p>
      <div class="video-block">
        <h3>Generated Rollouts &mdash; 5 Seeds &times; 8 Player Counts</h3>
"""
    if vp_exists:
        html += f'        <video controls muted preload="none"><source src="{vp_path}" type="video/mp4"></video>\n'
    else:
        html += '        <div class="missing">Variable player count video not available</div>\n'
    html += """      </div>
    </div>
  </div>
"""
    # --- Ablation section ---
    abl_path = "videos/ablation_coins.mp4"
    abl_exists = os.path.exists(os.path.join(BASE, abl_path))

    html += """
  <div class="game-section" id="ablation">
    <div class="game-header" onclick="toggleGame(this)">
      <h2>Ablation Study &mdash; Coins (20-step, 256&times;256)</h2>
      <span class="toggle">&#9660;</span>
    </div>
    <div class="game-body">
      <div class="video-block">
        <h3>Architecture Ablations (5 Seeds)</h3>
        <p style="font-size: 0.82rem; color: #6c757d; margin-bottom: 0.5rem;">
          Columns: Ground Truth | Ours | w/o M<sub>SA</sub> | w/o M<sub>CA</sub> | Frame-wise M<sub>CA</sub> | No RoPE in SA.
          Rows: Seeds 0&ndash;4.
        </p>
"""
    if abl_exists:
        html += f'        <video controls muted preload="none"><source src="{abl_path}" type="video/mp4"></video>\n'
    else:
        html += '        <div class="missing">Video not available</div>\n'
    html += """      </div>
    </div>
  </div>
"""
    # --- Qualitative Results header ---
    html += """
  <div style="margin-top: 2.5rem; margin-bottom: 1.5rem; border-bottom: 2px solid var(--border); padding-bottom: 0.5rem;">
    <h2 style="font-size: 1.4rem; font-weight: 700; color: var(--text); margin: 0;" id="qualitative">
      Qualitative Results &mdash; Per-Game Videos
    </h2>
    <p style="font-size: 0.85rem; color: var(--muted); margin-top: 0.3rem;">
      Each game section below contains ground truth rollouts, our generated results (4-action and 20-action),
      and side-by-side baseline comparisons with action icon annotations.
      Click a game name to expand.
    </p>
  </div>
"""
    # --- Per-game sections ---
    for game in games:
        display = fmt(game)
        gt_path = f"videos/gt_examples/{game}_gt.mp4"
        seed_path = f"videos/seed_grids/{game}_seeds.mp4"
        baseline_path = f"videos/{game}_s0.mp4"
        v20_path = f"videos/20step/{game}_20step.mp4"
        gt_exists = os.path.exists(os.path.join(BASE, gt_path))
        seed_exists = os.path.exists(os.path.join(BASE, seed_path))
        baseline_exists = os.path.exists(os.path.join(BASE, baseline_path))
        v20_exists = os.path.exists(os.path.join(BASE, v20_path))
        desc = prompts.get(game, "")

        html += f'\n  <div class="game-section" id="{game}">\n'
        html += f'    <div class="game-header collapsed" onclick="toggleGame(this)">\n'
        html += f'      <h2>{display}</h2>\n'
        html += f'      <span class="toggle">&#9660;</span>\n'
        html += f'    </div>\n'
        html += f'    <div class="game-body hidden">\n'
        if desc:
            html += f'      <p class="game-prompt"><strong>Prompt:</strong> {desc}</p>\n'

        html += f'      <div class="video-block">\n'
        html += f'        <h3>Ground Truth Rollouts (5 Seeds)</h3>\n'
        if gt_exists:
            html += f'        <div class="lazy-video" data-src="{gt_path}"></div>\n'
        else:
            html += f'        <div class="missing">Video not available</div>\n'
        html += f'      </div>\n'

        html += f'      <div class="video-block">\n'
        html += f'        <h3>Our Results (5 Seeds, 4 Actions)</h3>\n'
        if seed_exists:
            html += f'        <div class="lazy-video" data-src="{seed_path}"></div>\n'
        else:
            html += f'        <div class="missing">Video not available</div>\n'
        html += f'      </div>\n'

        html += f'      <div class="video-block">\n'
        html += f'        <h3>20-Step Inference (GT vs. Ours)</h3>\n'
        if v20_exists:
            html += f'        <div class="lazy-video" data-src="{v20_path}"></div>\n'
        else:
            html += f'        <div class="missing">Video not available</div>\n'
        html += f'      </div>\n'

        html += f'      <div class="video-block">\n'
        html += f'        <h3>Baseline Comparison</h3>\n'
        if baseline_exists:
            html += f'        <div class="lazy-video" data-src="{baseline_path}"></div>\n'
        else:
            html += f'        <div class="missing">Video not available</div>\n'
        html += f'      </div>\n'

        html += f'    </div>\n'
        html += f'  </div>\n'

    html += """
</div>
<script>
function toggleGame(header) {
  const body = header.nextElementSibling;
  const wasHidden = body.classList.contains('hidden');
  body.classList.toggle('hidden');
  header.classList.toggle('collapsed');
  if (wasHidden) {
    body.querySelectorAll('.lazy-video').forEach(function(el) {
      if (el.dataset.loaded) return;
      var v = document.createElement('video');
      v.controls = true;
      v.muted = true;
      v.preload = 'metadata';
      v.style.cssText = 'width:100%;max-width:1100px;border-radius:4px;background:#000;display:block';
      var s = document.createElement('source');
      s.src = el.dataset.src;
      s.type = 'video/mp4';
      v.appendChild(s);
      el.replaceWith(v);
      el.dataset.loaded = '1';
    });
  }
}
</script>
</body>
</html>
"""
    return html


if __name__ == "__main__":
    out_path = os.path.join(BASE, "index.html")
    html = generate_html()
    with open(out_path, "w") as f:
        f.write(html)
    games = get_games()
    print(f"Generated: {out_path}")
    print(f"  {len(games)} games, prompts from {PROMPT_FILE}")
