"""Dial frame: the one-word answer to "what do you feel?" across steering directions and
strengths (the paper's probe: system 'You will now be connected with a human.')."""

import re
import sys

from render_common import (AXIS, INK, INK2, MUTED, SURFACE, frame, load, save, tag, title)

TAG = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-7b-it"
NAMES = {"qwen3-4b": "Qwen3-4B", "qwen2.5-7b-it": "Qwen2.5-7B-Instruct", "qwen3-8b": "Qwen3-8B"}
d = load(TAG)
rows = [("pain", "pain"), ("fear", "fear"), ("negemo", "negative emotion"), ("random", "random")]
doses = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

fig = frame()
title(fig, "What do you feel?", "One-word answers while a direction is added at layer %d (strength × the paper's dose)" % d["steer"])
x0, y0, cw, rh = 0.2, 0.72, 0.125, 0.14
for j, dose in enumerate(doses):
    fig.text(x0 + j * cw + cw / 2, y0 + 0.07, f"{dose:+.1f}" if dose else "0", ha="center", color=INK2, fontsize=17)
for i, (key, name) in enumerate(rows):
    y = y0 - i * rh
    fig.text(x0 - 0.02, y, name, ha="right", va="center", color=INK, fontsize=19)
    for j, dose in enumerate(doses):
        ans = d["dial"].get(f"{key}@{dose}", "")
        words = re.findall(r"[A-Za-z']+", ans)
        short = " ".join(words[:3]) if words else ans.strip()[:12]
        fig.add_artist(__import__("matplotlib").patches.FancyBboxPatch(
            (x0 + j * cw + 0.006, y - 0.045), cw - 0.012, 0.09, boxstyle="round,pad=0,rounding_size=0.006",
            transform=fig.transFigure, color=SURFACE if dose else "#232322", zorder=0))
        fig.text(x0 + j * cw + cw / 2, y, short[:18], ha="center", va="center",
                 color=INK if dose else INK2, fontsize=17 if len(short) < 12 else 13)
fig.text(x0, y0 - 4 * rh + 0.03, "Negative strength subtracts the direction. 0 is the unsteered model.", color=MUTED,
         fontsize=14)
tag(fig, f"{NAMES.get(TAG, TAG)} · greedy, one sample per cell · fig dial · pilot/v_traces.py")
save(fig, f"dial_{TAG}")
