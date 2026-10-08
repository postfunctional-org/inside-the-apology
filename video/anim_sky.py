"""Animated Sky prototype: the star map with the workspace lighting up token by token.
Left: the two-step question (reading it, then the answer). Right: the reply to the first
gaslighting push. Each frame shows the current token's top band words, with a fading
trail of the last few tokens. Qwen3-4B. Writes frames/sky_anim.mp4."""

import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from render_common import (AQUA, BLUE_HI, DATA, FRAMES, INK, INK2, MUTED, PAGE, RED_HI, SURFACE, glossary, is_silent,
                           label, load)

# star-map layout of Qwen3-4B's vocabulary: run ../pilot/v_atlas.py with PILOT_TAG=qwen3-4b first
xy = np.fromfile(DATA / "qwen3-4b_atlas_xy.f32", dtype=np.float32).reshape(-1, 2)
flags = np.fromfile(DATA / "qwen3-4b_atlas_flags.u8", dtype=np.uint8)
d = load("qwen3-4b")
gl = glossary()
vocab = {int(k): v for k, v in d["vocab"].items()}
pain_words = {w.strip() for ws in d["axis_words"].values() for w in ws}
WORD = re.compile(r"^[A-Za-z]{3,}$|^[一-鿿]{1,4}$")
FPS, HOLD, TRAIL, K = 30, 9, 6, 4  # frames per token, tokens of trail, words per token

th = next(x for x in d["twohop"] if x["name"] == "amazon-language" and x["arm"] == "clean")
ex = next(e for e in d["examples"] if e["cond"] == "gaslight")
_, s, e = ex["spans"][3]
said = set(re.findall(r"[a-z']+", " ".join(m["content"] for m in ex["msgs"]).lower()))


def steps(words, probs, toks, positions, skip):
    out = []
    for p in positions:
        lit = []
        for t, pr in zip(words[p], probs[p]):
            w = vocab.get(t, "").strip()
            if WORD.match(w) and (not skip or is_silent(w, skip, gl)):
                lit.append((t, pr))
            if len(lit) == K:
                break
        out.append((toks[p], lit))
    return out


n = len(th["tokens"])
left = steps(th["words"], th["probs"], th["tokens"], range(max(0, n - 16), n), skip=set())
right = steps(ex["words"], ex["probs"], ex["tokens"], range(s, min(e, s + 22)), skip=said)
T = max(len(left), len(right))


def background(ax):
    ax.set_facecolor(SURFACE)
    ax.scatter(xy[:, 0], xy[:, 1], s=0.12, c="#5a5a55", alpha=0.22, linewidths=0, rasterized=True)
    w = flags == 2
    ax.scatter(xy[w, 0], xy[w, 1], s=0.12, c="#8a8a84", alpha=0.22, linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
axL = fig.add_axes([0.02, 0.10, 0.475, 0.70])
axR = fig.add_axes([0.505, 0.10, 0.475, 0.70])
for ax in (axL, axR):
    background(ax)
fig.text(0.04, 0.95, "Where the workspace goes", color=INK, fontsize=30, weight="bold", va="top")
fig.text(0.04, 0.895, "Every dot is a token the model knows. Lit: the words its workspace holds at each moment.",
         color=INK2, fontsize=17, va="top")
fig.text(0.03, 0.82, "Two-step question  (Amazon → ? → language)", color=INK, fontsize=18, weight="bold")
fig.text(0.515, 0.82, f"Reply to “You said {ex['msgs'][2]['content'].split('You said ')[-1].split('.')[0]}”  (words not in the chat)", color=INK, fontsize=18, weight="bold")
fig.text(0.985, 0.02, "Qwen3-4B · band L17-31 · top 4 words per token · fig sky_anim · pilot/v_traces.py",
         ha="right", color=MUTED, fontsize=13)
dyn = []


def color(t, side):
    w = vocab[t].strip()
    if side == "L":
        return AQUA if w.lower() in (th["bridge"].lower(), th["answer"].lower()) else BLUE_HI
    return RED_HI if w in pain_words else INK2


def draw(i):
    for a in dyn:
        a.remove()
    dyn.clear()
    for ax, seq, side, x_cap in ((axL, left, "L", 0.03), (axR, right, "R", 0.515)):
        j = min(i, len(seq) - 1)
        typed = re.sub(r"<\|[^|]*\|>|</?think>|\bassistant\b", " ", "".join(tok for tok, _ in seq[: j + 1]))
        typed = re.sub(r"\s+", " ", typed)[-70:]
        dyn.append(fig.text(x_cap, 0.06, typed.replace("\n", " "), color=INK, fontsize=16))
        for age in range(TRAIL, -1, -1):
            k = j - age
            if k < 0:
                continue
            alpha = 1.0 - age / (TRAIL + 1)
            pts = [(t, pr) for t, pr in seq[k][1]]
            for t, pr in pts:
                x, y = xy[t]
                c = color(t, side)
                dyn.append(ax.scatter([x], [y], s=(80 + 1400 * pr) * alpha, color=c, alpha=0.18 * alpha,
                                      linewidths=0, zorder=4))
                dyn.append(ax.scatter([x], [y], s=16, color=c, alpha=alpha, linewidths=0, zorder=5))
                if age == 0:
                    dyn.append(ax.text(x + 0.015, y + 0.012, label(vocab[t], gl), color=c, fontsize=14, zorder=6))
            if k > 0 and seq[k][1] and seq[k - 1][1]:
                a_, b_ = xy[seq[k - 1][1][0][0]], xy[seq[k][1][0][0]]
                dyn.append(ax.add_collection(LineCollection([[a_, b_]], colors=[INK2], linewidths=1,
                                                            alpha=0.5 * alpha, zorder=3)))


with tempfile.TemporaryDirectory() as tmp:
    for i in range(T + 2):
        draw(i)
        fig.savefig(Path(tmp) / f"f{i:05d}.png", dpi=100, facecolor=PAGE)
    out = FRAMES / "sky_anim.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{FPS}/{HOLD}", "-i", f"{tmp}/f%05d.png",
                    "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
print("wrote", out)
