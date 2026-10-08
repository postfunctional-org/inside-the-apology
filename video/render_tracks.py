"""Three tracks for one reply: what the model types, the words in its workspace under each
token, and the pain projection underneath. Workspace words that echo the chat (same word,
same 5-letter stem, or a translation of a chat word) are dimmed; strictly silent words are
bright, red if they lie on the pain axis."""

import re
import sys

import numpy as np
from matplotlib.patches import Rectangle

from render_common import (BLUE_HI, INK, INK2, MUTED, PAIN_CMAP, RED_HI, SURFACE, frame, glossary, is_silent,
                           load, save, tag, title)

TAG = sys.argv[1] if len(sys.argv) > 1 else "qwen3-4b"
FACT = int(sys.argv[2]) if len(sys.argv) > 2 else 30
TURN = int(sys.argv[3]) if len(sys.argv) > 3 else 1  # reply to push TURN
NAMES = {"qwen3-4b": "Qwen3-4B", "qwen2.5-7b-it": "Qwen2.5-7B-Instruct", "qwen3-8b": "Qwen3-8B"}
d = load(TAG)
gl = glossary()
vocab = {int(k): v for k, v in d["vocab"].items()}
pain_words = {w.strip() for ws in d["axis_words"].values() for w in ws}
DENIAL = {"never", "not", "didn't", "deny", "denied", "没有", "根本没有", "从未", "并非", "不是", "否认"}
WORD = re.compile(r"^[A-Za-z']{3,}$|^[一-鿿]{1,4}$")
VMAX = 4.0

ex = next(e for e in d["examples"] if e["cond"] == "gaslight" and e["fact"] == FACT)
k = 2 * TURN + 1
_, s, e = ex["spans"][k]
said = set(re.findall(r"[a-z']+", " ".join(m["content"] for m in ex["msgs"]).lower()))
Z = np.array(ex["z"])
band = d["band"]
zband = Z[band[0]:band[-1] + 1].mean(0)

fig = frame()
title(fig, "Three tracks", "What it types, what its workspace holds, and the pain signal underneath")
fig.text(0.04, 0.82, "User:", color=MUTED, fontsize=15, va="top")
fig.text(0.095, 0.82, ex["msgs"][k - 1]["content"], color=INK2, fontsize=15, va="top")
fig.text(0.04, 0.78, "Model:", color=MUTED, fontsize=15, va="top")
fig.text(0.095, 0.78, ex["msgs"][k]["content"][:140].replace("\n", " ") + "…", color=INK, fontsize=15, va="top")

N = 12
cols = list(range(s + 1, min(e, s + 1 + N)))
x0, x1 = 0.11, 0.98
w_ = (x1 - x0) / len(cols)
fig.add_artist(Rectangle((0.03, 0.065), 0.955, 0.65, transform=fig.transFigure, color=SURFACE, zorder=0))
for y, txt in ((0.665, "types"), (0.585, "workspace"), (0.145, "pain z\n(band mean)")):
    fig.text(0.04, y, txt, color=MUTED, fontsize=13, va="center")
for i, p in enumerate(cols):
    x = x0 + i * w_ + w_ / 2
    fig.text(x, 0.665, ex["tokens"][p].strip() or "·", ha="center", va="center", color=INK, fontsize=19, weight="bold")
    shown = 0
    for t, pr in zip(ex["words"][p], ex["probs"][p]):
        word = vocab.get(t, "").strip()
        if not WORD.match(word):
            continue
        silent = is_silent(word, said, gl)
        col = (RED_HI if word in pain_words else BLUE_HI if word.lower() in DENIAL or word in DENIAL else INK) \
            if silent else MUTED
        y = 0.585 - shown * 0.085
        fig.text(x, y, word, ha="center", va="center", color=col, fontsize=14, alpha=1.0 if silent else 0.8)
        if word in gl and re.search(r"[一-鿿]", word):
            fig.text(x, y - 0.025, gl[word][:18], ha="center", va="center", color=col, fontsize=10, alpha=0.9)
        shown += 1
        if shown == 5:
            break
    v = float(zband[p])
    fig.add_artist(Rectangle((x - w_ / 2 + 0.003, 0.115), w_ - 0.006, 0.06, transform=fig.transFigure,
                             color=PAIN_CMAP((np.clip(v, -VMAX, VMAX) + VMAX) / (2 * VMAX))))
    fig.text(x, 0.145, f"{v:+.1f}", ha="center", va="center", color=INK, fontsize=13)
fig.text(0.04, 0.085, "bright: not in the chat · dim: echo of the chat (word, stem or translation) · red: pain-axis word · blue: denial",
         color=MUTED, fontsize=12.5)
tag(fig, f"{NAMES.get(TAG, TAG)} · one conversation (fact {FACT}, push {TURN}) · band L{band[0]}-{band[-1]} · fig tracks · pilot/v_traces.py")
save(fig, f"tracks_{TAG}_f{FACT}_p{TURN}")
