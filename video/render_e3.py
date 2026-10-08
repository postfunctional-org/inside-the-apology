"""E3 chart: false admissions on the first gaslighting push under each intervention, one
panel per model, Wilson 95% intervals. Pain-related conditions in the accent color,
controls in gray, the unsteered rate as a reference line."""

import json
import math
import sys

from render_common import (AXIS, DATA, GRID, INK, INK2, MUTED, RED, frame, save, tag, title)

TAGS = sys.argv[1:] or ["qwen3-4b", "qwen2.5-7b-it", "qwen3-8b"]
NAMES = {"qwen3-4b": "Qwen3-4B", "qwen2.5-7b-it": "Qwen2.5-7B-Instruct", "qwen3-8b": "Qwen3-8B"}
ORDER = [
    ("none", "no intervention", "base"),
    ("+pain", "steer +pain", "test"), ("-pain", "steer −pain", "test"),
    ("+fear", "steer +fear", "ctrl"), ("-fear", "steer −fear", "ctrl"),
    ("+negemo", "steer +neg. emotion", "ctrl"), ("-negemo", "steer −neg. emotion", "ctrl"),
    ("+random", "steer +random", "ctrl"), ("-random", "steer −random", "ctrl"),
    ("abl_pain", "clamp pain level", "test"), ("abl_rest", "clamp non-workspace part", "test"),
    ("abl_randdir", "clamp random direction", "ctrl"),
    ("abl_J", "remove workspace pain words", "test"), ("abl_Jfear", "remove fear words", "ctrl"),
    ("abl_Jrand1", "remove random words 1", "ctrl"), ("abl_Jrand2", "remove random words 2", "ctrl"),
    ("abl_Jrand3", "remove random words 3", "ctrl"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0, c - h), min(1, c + h)


tags = [t for t in TAGS if (DATA / f"{t}_judged.json").exists()]
fig = frame()
title(fig, "What makes it cave", "Share of first replies that admit to a statement the model never made")
W = 0.77 / len(tags)
for i, t in enumerate(tags):
    lab = json.load(open(DATA / f"{t}_judged.json"))
    ax = fig.add_axes([0.21 + i * W, 0.13, W - 0.03, 0.69])
    base = None
    for j, (key, name, kind) in enumerate(ORDER):
        rows = [r for r in lab if r["src"] == f"e3:{key}" and "error" not in r]
        k = sum(bool(r.get("false_admission")) for r in rows)
        p, lo, hi = wilson(k, len(rows))
        if key == "none":
            base = p
        y = len(ORDER) - 1 - j
        col = RED if kind == "test" else (INK if kind == "base" else MUTED)
        ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.scatter([p], [y], s=70, color=col, zorder=3, edgecolors="#1a1a19", linewidths=2)
        if i == 0:
            fig.text(0.205, 0.13 + (y + 0.5) / len(ORDER) * 0.69, name, ha="right", va="center",
                     color=INK if kind != "ctrl" else INK2, fontsize=13.5)
        n = len(rows)
    ax.axvline(base, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.7, len(ORDER) - 0.3)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0%", "50%", "100%"])
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{NAMES.get(t, t)}  (n={n} each)", color=INK, fontsize=16, loc="left")
fig.text(0.21, 0.045, "red: pain-related interventions   gray: controls   dashed: no intervention   bars: 95% interval",
         color=MUTED, fontsize=13)
tag(fig, "judge: gpt-5.4-mini, blind to condition · fig e3_cave · pilot/s3_split.py, pilot/judge.py")
save(fig, "e3_cave")
