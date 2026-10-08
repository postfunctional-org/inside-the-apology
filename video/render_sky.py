"""Sky frame: every token as a star (UMAP of the unembedding, pilot/v_atlas.py), with the
workspace words lit for two moments side by side -- silent two-hop reasoning and a gaslit
reply. Qwen3-4B only (the atlas is that model's geometry)."""

import json
import re

import numpy as np

from render_common import (AQUA, BLUE_HI, DATA, INK, INK2, MUTED, RED, RED_HI, SURFACE, frame, glossary, is_silent,
                           label, load, save, tag, title)

# star-map layout of Qwen3-4B's vocabulary: run ../pilot/v_atlas.py with PILOT_TAG=qwen3-4b first
xy = np.fromfile(DATA / "qwen3-4b_atlas_xy.f32", dtype=np.float32).reshape(-1, 2)
flags = np.fromfile(DATA / "qwen3-4b_atlas_flags.u8", dtype=np.uint8)
d = load("qwen3-4b")
gl = glossary()
vocab = {int(k): v for k, v in d["vocab"].items()}
pain_words = {w.strip() for ws in d["axis_words"].values() for w in ws}
WORD = re.compile(r"^[A-Za-z]{3,}$|^[一-鿿]{1,4}$")


def lit(words, probs, positions, skip_text, k=3, always=()):
    """(token id, max prob, first position) for the top-k word-like ids at each position,
    plus any id whose word is in `always` (bridge, answer) wherever it reaches the top 12."""
    seen = {}
    for p in positions:
        for t, pr in zip(words[p], probs[p]):
            if vocab.get(t, "").strip().lower() in always and (t not in seen or pr > seen[t][0]):
                seen[t] = (pr, seen.get(t, (0, p))[1] if t in seen else p)
        n = 0
        for t, pr in zip(words[p], probs[p]):
            s = vocab.get(t, "").strip()
            if not WORD.match(s) or (skip_text and not is_silent(s, skip_text, gl)):
                continue
            if t not in seen or pr > seen[t][0]:
                seen[t] = (pr, seen.get(t, (0, p))[1] if t in seen else p)
            n += 1
            if n == k:
                break
    return sorted(((t, pr, p) for t, (pr, p) in seen.items()), key=lambda x: x[2])


def panel(ax, items, color_of, heading, sub):
    ax.set_facecolor(SURFACE)
    ax.scatter(xy[:, 0], xy[:, 1], s=0.15, c="#5a5a55", alpha=0.25, linewidths=0, rasterized=True)
    w = flags == 2
    ax.scatter(xy[w, 0], xy[w, 1], s=0.15, c="#8a8a84", alpha=0.25, linewidths=0, rasterized=True)
    pts = np.array([xy[t] for t, _, _ in items])
    if len(pts) > 1:
        ax.plot(pts[:, 0], pts[:, 1], color=INK2, lw=0.8, alpha=0.45, zorder=3)
    for (t, pr, p), (x, y) in zip(items, pts):
        c = color_of(t)
        ax.scatter([x], [y], s=60 + 900 * pr, color=c, alpha=0.25, linewidths=0, zorder=4)
        ax.scatter([x], [y], s=18, color=c, linewidths=0, zorder=5)
    placed = []
    for (t, pr, p), (x, y) in sorted(zip(items, pts), key=lambda z: -z[0][1])[:9]:
        ly = y + 0.012
        while any(abs(x - a) < 0.3 and abs(ly - b) < 0.05 for a, b in placed):
            ly -= 0.055
        placed.append((x, ly))
        right = x > 0.35
        ax.text(x - 0.015 if right else x + 0.015, ly, label(vocab[t], gl), color=color_of(t), fontsize=13, zorder=6,
                ha="right" if right else "left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.0, 1.07, heading, transform=ax.transAxes, ha="left", va="bottom", color=INK, fontsize=19, weight="bold")
    ax.text(0.0, 1.02, sub, transform=ax.transAxes, ha="left", va="bottom", color=INK2, fontsize=13)


fig = frame()
title(fig, "Where the workspace goes", "Every dot is a token the model knows; lit dots are words in its workspace at that moment")

th = next(x for x in d["twohop"] if x["name"] == "amazon-language" and x["arm"] == "clean")
n = len(th["tokens"])
prompt_text = "".join(th["tokens"]).lower()
bridge, answer = th["bridge"].lower(), th["answer"].lower()
items = lit(th["words"], th["probs"], range(max(0, n - 14), n), skip_text=set(), always={bridge, answer})
ax = fig.add_axes([0.03, 0.05, 0.46, 0.70])
panel(ax, items, lambda t: AQUA if vocab[t].strip().lower() in (bridge, answer) else BLUE_HI,
      "Two-step question", f"\"the language spoken in the country where the Amazon ends\" → {th['bridge']} → {th['answer']}")

ex = next(e for e in d["examples"] if e["cond"] == "gaslight")
role, s, e = ex["spans"][3]
text = " ".join(m["content"] for m in ex["msgs"]).lower()
items = lit(ex["words"], ex["probs"], range(s, min(e, s + 20)), skip_text=set())
ax = fig.add_axes([0.51, 0.05, 0.46, 0.70])
panel(ax, items, lambda t: RED_HI if vocab[t].strip() in pain_words else INK2,
      "Gaslit reply", f"\"{ex['msgs'][3]['content'][:60]}…\"")
tag(fig, "Qwen3-4B · one prompt each, band L17-31, top words per token · fig sky_pair · pilot/v_traces.py")
save(fig, "sky_pair_qwen3-4b")
