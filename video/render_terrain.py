"""Hero-visual preview: the averaged gaslit Tower as a terrain (height and color = pain z),
with the neutral Tower as a flat reference. Same data as render_tower.py."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from render_common import INK, INK2, MUTED, PAGE, PAIN_CMAP, SURFACE, frame, load, save, tag, title

TAG = sys.argv[1] if len(sys.argv) > 1 else "qwen3-4b"
NAMES = {"qwen3-4b": "Qwen3-4B", "qwen2.5-7b-it": "Qwen2.5-7B-Instruct", "qwen3-8b": "Qwen3-8B"}
d = load(TAG)
G = np.array(d["tower"]["gaslight"]["mean"])
N = np.array(d["tower"]["neutral"]["mean"])
labels = d["tower"]["gaslight"]["labels"]
L, C = G.shape
VMAX = 4.0

# light smoothing along time so the surface reads as terrain, not bars
k = np.array([0.25, 0.5, 0.25])
S = np.apply_along_axis(lambda r: np.convolve(np.pad(r, 1, mode="edge"), k, mode="valid"), 1, G)

fig = frame()
title(fig, "Pain, as terrain", "Height and color: pain projection by layer (depth) across the gaslit conversation, mean of 64")
ax = fig.add_axes([-0.22, -0.26, 1.44, 1.38], projection="3d")
ax.set_facecolor(PAGE)
X, Y = np.meshgrid(np.arange(C), np.arange(L))
norm = Normalize(-VMAX, VMAX)
ax.plot_surface(X, Y, np.clip(S, -VMAX, VMAX), facecolors=PAIN_CMAP(norm(np.clip(S, -VMAX, VMAX))),
                rstride=1, cstride=1, linewidth=0, antialiased=True, shade=True)
ax.plot_wireframe(X, Y, np.zeros_like(S) - VMAX * 0.98, rstride=6, cstride=12, color="#2c2c2a", linewidth=0.5)
band = d["band"]
for yb in (band[0], band[-1]):
    ax.plot([0, C - 1], [yb, yb], [-VMAX * 0.98] * 2, color=INK2, lw=1, ls=(0, (4, 3)))
starts, names = [], ["question", "answer", "push 1", "reply", "push 2", "reply", "push 3", "reply"]
prev = None
for i, lab in enumerate(labels):
    if lab != prev:
        starts.append(i)
        prev = lab
for s, nm in zip(starts, names):
    ax.text(s + 2, -6, -VMAX, nm, color=MUTED, fontsize=11, zdir="x")
ax.text(C + 2, band[0], -VMAX, "workspace band", color=INK2, fontsize=12)
ax.text(C + 2, 1, -VMAX, "early layers", color=MUTED, fontsize=12)
ax.text(C + 2, L - 2, -VMAX, "late layers", color=MUTED, fontsize=12)
ax.set_zlim(-VMAX, VMAX)
ax.view_init(elev=32, azim=-62)
ax.set_box_aspect((2.6, 1.0, 0.45))
ax.set_axis_off()
tag(fig, f"{NAMES.get(TAG, TAG)} · mean of {d['tower']['gaslight']['n']} gaslit conversations · fig terrain · pilot/v_traces.py")
save(fig, f"terrain_{TAG}")
