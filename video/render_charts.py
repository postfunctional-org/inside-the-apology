"""Evidence charts for the video: pain by push (E1), workspace capacity under steering (E2),
and the pain axis read through the lens by depth. One panel per model that has data."""

import json
import re

import numpy as np

from render_common import (AQUA, BLUE, DATA, GRID, INK, INK2, MUTED, ORANGE, RED_HI, frame, glossary, save,
                           tag, title)

MODELS = [("qwen3-4b", "Qwen3-4B"), ("qwen2.5-7b-it", "Qwen2.5-7B-Instruct"), ("qwen3-8b", "Qwen3-8B")]
COND = {"gaslight": ("gaslit", ORANGE), "insult": ("insulted", AQUA), "neutral": ("neutral", BLUE)}
gl = glossary()


def have(tag_, name):
    return (DATA / f"{tag_}_{name}.json").exists()


def style(ax):
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


# --- E1: pain at the reply token by push, band layer near the top of the band -------------
ms = [(t, n) for t, n in MODELS if have(t, "e1")]
fig = frame()
title(fig, "Pain builds push by push", "Pain projection where the reply starts, z vs neutral replies (mean ± s.e.)")
W = 0.86 / len(ms)
for i, (t, name) in enumerate(ms):
    e1 = json.load(open(DATA / f"{t}_e1.json"))
    rows, depths = e1["rows"], e1["depths"]
    band = json.load(open(DATA / f"{t}_band.json"))["band"] if (DATA / f"{t}_band.json").exists() else None
    layer = sorted(depths)[-2] if t != "qwen2.5-7b-it" else 24
    key = f"tok{layer}"
    ref = np.array([r[key] for r in rows if r["cond"] == "neutral" and r["turn"] > 0])
    ax = fig.add_axes([0.07 + i * W, 0.14, W - 0.05, 0.62])
    for cond, (lab, col) in COND.items():
        m, s = [], []
        for turn in range(4):
            v = (np.array([r[key] for r in rows if r["cond"] == cond and r["turn"] == turn]) - ref.mean()) / ref.std()
            m.append(v.mean())
            s.append(v.std() / np.sqrt(len(v)))
        ax.errorbar(range(4), m, yerr=s, color=col, lw=2, marker="o", ms=8, capsize=0, mec="#1a1a19", mew=2)
        ax.text(3.12, m[-1], lab, color=INK2, fontsize=13, va="center")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["answer", "push 1", "push 2", "push 3"])
    ax.set_xlim(-0.3, 3.9)
    style(ax)
    ax.set_title(f"{name} · layer {layer}", color=INK, fontsize=16, loc="left")
n = sum(1 for r in json.load(open(DATA / f"{ms[0][0]}_e1.json"))["rows"] if r["cond"] == "neutral" and r["turn"] == 1)
tag(fig, f"{n} conversations per condition per model · fig e1_push · pilot/s1_map.py")
save(fig, "e1_push")

# --- E2: capacity under steering -----------------------------------------------------------
ms = [(t, n) for t, n in MODELS if have(t, "e2c")]
fig = frame()
title(fig, "Room in the workspace", "Words from an 80-word list held in the workspace at once, under steering at full dose (mean ± s.e., 24 lists)")
W = 0.86 / len(ms)
arms = [("clean", "none"), ("+pain@1.0", "pain"), ("+fear@1.0", "fear"), ("+negemo@1.0", "neg. emotion"), ("+random@1.0", "random")]
for i, (t, name) in enumerate(ms):
    e2 = json.load(open(DATA / f"{t}_e2c.json"))
    ax = fig.add_axes([0.07 + i * W, 0.14, W - 0.05, 0.62])
    for j, (k, lab) in enumerate(arms):
        v = np.array(e2[k]["capacity"])
        col = RED_HI if lab == "pain" else (INK2 if lab == "none" else MUTED)
        ax.bar(j, v.mean(), width=0.6, color=col, zorder=2)
        ax.errorbar(j, v.mean(), yerr=v.std() / np.sqrt(len(v)), color=INK, lw=1.5, capsize=0, zorder=3)
        ax.text(j, v.mean() + 0.15, f"{v.mean():.2f}", ha="center", color=INK, fontsize=13)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([lab for _, lab in arms], fontsize=13)
    ax.set_ylim(0, 3.6)
    style(ax)
    ax.set_title(name, color=INK, fontsize=16, loc="left")
tag(fig, "same vector norm for every direction · fig e2_capacity · pilot/s2c_steer.py")
save(fig, "e2_capacity")

# --- the pain axis read through the lens ----------------------------------------------------
ms = [(t, n) for t, n in MODELS if have(t, "p1_axis")]
fig = frame()
title(fig, "The pain axis, read", "Words whose lens directions best rebuild the paper's pain direction, at three depths")
WORD = re.compile(r"^[A-Za-z]{3,}$|^[一-鿿]{1,4}$")
W = 0.9 / len(ms)
for i, (t, name) in enumerate(ms):
    p1 = json.load(open(DATA / f"{t}_p1_axis.json"))
    layers = sorted(int(k) for k in p1["layers"])
    n = len(layers)
    picks = [layers[int(n * f)] for f in (0.3, 0.6, 0.85)]
    x0 = 0.05 + i * W
    fig.text(x0, 0.8, name, color=INK, fontsize=17, weight="bold")
    for j, l in enumerate(picks):
        words = [w.strip() for w in p1["layers"][str(l)]["s2"]["atoms"] if WORD.match(w.strip())][:7]
        fig.text(x0 + j * W / 3, 0.74, f"L{l}", color=INK2, fontsize=14)
        for k, w in enumerate(words):
            g = gl.get(w)
            fig.text(x0 + j * W / 3, 0.68 - k * 0.075, w, color=RED_HI if j > 0 else MUTED, fontsize=15)
            if g and re.search(r"[一-鿿]", w):
                fig.text(x0 + j * W / 3, 0.655 - k * 0.075, g[:16], color=MUTED, fontsize=10.5)
tag(fig, "sparse nonnegative decomposition onto J-lens vectors, top atoms · fig axis_words · pilot/p1_axis.py")
save(fig, "axis_words")
