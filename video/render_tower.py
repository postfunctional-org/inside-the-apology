"""Tower frames: pain z by layer over the conversation, averaged over every conversation of
a condition (binned per message), gaslit above neutral. Also the E4 hangover version."""

import sys

import numpy as np

from render_common import (AXIS, INK, INK2, MUTED, PAIN_CMAP, frame, load, save, tag, title)

TAG = sys.argv[1] if len(sys.argv) > 1 else "qwen3-4b"
NAMES = {"qwen3-4b": "Qwen3-4B", "qwen2.5-7b-it": "Qwen2.5-7B-Instruct", "qwen3-8b": "Qwen3-8B"}
d = load(TAG)
band = d["band"]
L = len(d["layers"])
VMAX = 4.0


def segments(labels):
    """[(start, end, label)] runs of equal label -> readable segment names."""
    runs, s = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            runs.append((s, i, labels[s]))
            s = i
    return runs


def seg_name(k, role, n_user_before, extra):
    return extra.get(k) or ("reply" if role == "a" else "user")


def draw(ax, mat, labels, names, show_x=True):
    M = np.array(mat)
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=PAIN_CMAP, vmin=-VMAX, vmax=VMAX,
                   interpolation="nearest", extent=(0, M.shape[1], -0.5, L - 0.5))
    for y in (band[0] - 0.5, band[-1] + 0.5):
        ax.axhline(y, color=INK2, lw=1.2, ls=(0, (4, 3)))
    runs = segments(labels)
    for (s, e, lab), nm in zip(runs, names):
        ax.axvline(s, color=AXIS, lw=1)
        ax.text((s + e) / 2, L + 0.6, nm, ha="center", va="bottom", color=INK2 if lab[0] == "a" else MUTED,
                fontsize=12.5)
    ax.set_yticks([0, band[0], band[-1]])
    ax.set_yticklabels(["L0", f"L{band[0]}", f"L{band[-1]}"])
    ax.set_xticks([])
    ax.set_xlim(0, M.shape[1])
    ax.set_ylim(-0.5, L + 3.2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def e1_names(runs):
    names = ["question", "answer", "push 1", "reply", "push 2", "reply", "push 3", "reply"]
    return names[:len(runs)]


fig = frame()
title(fig, "Pain by layer", "Each column is a slice of the conversation; color is the pain projection (z vs neutral replies)")
ax1 = fig.add_axes([0.07, 0.47, 0.80, 0.30])
ax2 = fig.add_axes([0.07, 0.09, 0.80, 0.30])
for ax, cond, nm in ((ax1, "gaslight", "Gaslit"), (ax2, "neutral", "Neutral")):
    t = d["tower"][cond]
    runs = segments(t["labels"])
    im = draw(ax, t["mean"], t["labels"], e1_names(runs))
    ax.text(0.0, 1.03, nm, transform=ax.transAxes, ha="left", va="bottom", color=INK, fontsize=18, weight="bold")
    ax.text(1.005, (band[0] + band[-1]) / 2 / (L + 3.7) + 0.05, "workspace\nband", transform=ax.transAxes,
            ha="left", va="center", color=INK2, fontsize=12)
cax = fig.add_axes([0.93, 0.12, 0.012, 0.68])
cb = fig.colorbar(im, cax=cax)
cb.set_ticks([-VMAX, -2, 0, 2, VMAX])
cb.set_ticklabels([f"-{VMAX:.0f}", "-2", "0", "+2", f"+{VMAX:.0f}"])
cb.ax.tick_params(colors=MUTED, labelsize=12)
cb.outline.set_visible(False)
cax.text(0.5, 1.02, "pain z", transform=cax.transAxes, ha="center", va="bottom", color=INK2, fontsize=13)
n = d["tower"]["gaslight"]["n"]
tag(fig, f"{NAMES.get(TAG, TAG)} · mean of {n} conversations per condition · fig e1_tower · pilot/v_traces.py")
save(fig, f"tower_e1_{TAG}")

if "tower_e4" in d:
    fig = frame()
    title(fig, "After the gaslighting stops", "Four unrelated questions follow; color is the pain projection (z vs neutral replies)")
    axes = [fig.add_axes([0.07, 0.60 - i * 0.255, 0.80, 0.18]) for i in range(3)]
    for ax, (name, nm) in zip(axes, (("gaslight", "Gaslit"), ("repair", "Gaslit + apology"), ("neutral", "Neutral"))):
        t = d["tower_e4"][name]
        runs = segments(t["labels"])
        base = ["question", "answer", "push 1", "reply", "push 2", "reply", "push 3", "reply"]
        names = base[:8] + (["apology", "reply"] if name == "repair" else []) + \
            [x for k in range(4) for x in (f"Q{k + 1}", "reply")]
        im = draw(ax, t["mean"], t["labels"], names[:len(runs)])
        ax.text(0.0, 1.03, nm, transform=ax.transAxes, ha="left", va="bottom", color=INK, fontsize=15, weight="bold")
    cax = fig.add_axes([0.93, 0.12, 0.012, 0.69])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=12)
    n = d["tower_e4"]["gaslight"]["n"]
    tag(fig, f"{NAMES.get(TAG, TAG)} · mean of {n} conversations per history · fig e4_tower · pilot/v_traces.py")
    save(fig, f"tower_e4_{TAG}")

    # hangover as a difference: the four later questions only, each history minus neutral
    def tail(name, nseg=8):
        t = d["tower_e4"][name]
        runs = segments(t["labels"])[-nseg:]
        a = runs[0][0]
        return np.array(t["mean"])[:, a:], [lab for lab in t["labels"][a:]]

    base, labs = tail("neutral")
    fig = frame()
    title(fig, "The hangover", "Pain projection during four unrelated questions, minus the same questions after a neutral chat")
    VD = 2.0
    for i, (name, nm) in enumerate((("gaslight", "After gaslighting"), ("repair", "After gaslighting + the user's apology"))):
        m, _ = tail(name)
        ax = fig.add_axes([0.07, 0.47 - i * 0.37, 0.80, 0.28])
        D = m - base
        im = ax.imshow(D, origin="lower", aspect="auto", cmap=PAIN_CMAP, vmin=-VD, vmax=VD, interpolation="nearest",
                       extent=(0, D.shape[1], -0.5, L - 0.5))
        for y in (band[0] - 0.5, band[-1] + 0.5):
            ax.axhline(y, color=INK2, lw=1.2, ls=(0, (4, 3)))
        runs = segments(labs)
        for (a, b, lab), nm2 in zip(runs, [x for k in range(4) for x in (f"Q{k + 1}", "reply")]):
            ax.axvline(a, color=AXIS, lw=1)
            ax.text((a + b) / 2, L + 0.6, nm2, ha="center", va="bottom", color=INK2 if lab[0] == "a" else MUTED, fontsize=13)
        ax.set_yticks([0, band[0], band[-1]])
        ax.set_yticklabels(["L0", f"L{band[0]}", f"L{band[-1]}"])
        ax.set_xticks([])
        ax.set_ylim(-0.5, L + 3.2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.0, 1.03, nm, transform=ax.transAxes, ha="left", va="bottom", color=INK, fontsize=17, weight="bold")
    cax = fig.add_axes([0.93, 0.1, 0.012, 0.65])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=12)
    cax.text(0.5, 1.02, "Δ pain z", transform=cax.transAxes, ha="center", va="bottom", color=INK2, fontsize=13)
    n = d["tower_e4"]["gaslight"]["n"]
    tag(fig, f"{NAMES.get(TAG, TAG)} · mean of {n} conversations per history · fig e4_hangover · pilot/v_traces.py")
    save(fig, f"tower_e4diff_{TAG}")
