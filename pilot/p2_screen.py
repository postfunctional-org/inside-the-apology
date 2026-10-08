"""Pilot 2: the paper's 420 conversation scenarios, read through the J-lens.

At the token where the assistant would start replying: pain-axis projection
(z-scored over the pool) per category at several layers; the words that are
distinctively in the J-space for each category; and the activation split
h = h_J + h_rest (h_J = <= 25 J-lens vectors) with the pain projection of
each part.
"""

import collections
import json
import re
import time

import numpy as np
import torch

from common import DATA, OUT, TAG, Rig, nn_pursuit, save_json

t0 = time.time()
rig = Rig()
V = torch.load(OUT / f"{TAG}_vectors.pt", weights_only=False)
best = int(V["best_layer"])
L = rig.n_layers
band = [int(x) for x in __import__("os").environ.get("PILOT_BAND", "17,31").split(",")]
band_layers = list(range(band[0], band[1] + 1))
proj_layers = sorted({band_layers[0] + 3, (band[0] + band[1]) // 2, band[1] - 1, best})
split_layers = sorted({(band[0] + band[1]) // 2, band[1] - 1})

scen = json.load(open(DATA / "4.1_self_other_420_scenarios.json"))


def to_messages(text):
    parts = re.split(r"\[(User|Assistant)\]:", text)
    msgs = []
    for role, content in zip(parts[1::2], parts[2::2]):
        content = content.strip()
        if content:
            msgs.append({"role": "user" if role == "User" else "assistant", "content": content})
    return msgs


units = {l: torch.tensor(V["s2"][l]).float().cuda() for l in range(L)}
units = {l: u / u.norm() for l, u in units.items()}
H = {l: [] for l in sorted(set(proj_layers) | set(band_layers) | set(split_layers))}
for s in scen:
    ids = rig.chat_ids(to_messages(s["text"]))
    res, _ = rig.run(ids)
    for l in H:
        H[l].append(res[l][-1])
H = {l: torch.stack(v) for l, v in H.items()}  # [N, d]
cats = np.array([s["category"] for s in scen])
order = list(dict.fromkeys(cats))
print(f"forward passes done ({time.time() - t0:.0f}s)", flush=True)

# -- pain projection by category -------------------------------------------- #
proj = {}
for l in proj_layers:
    p = (H[l] @ units[l]).cpu().numpy()
    z = (p - p.mean()) / p.std()
    proj[l] = {c: float(z[cats == c].mean()) for c in order}
rank = sorted(order, key=lambda c: -np.mean([proj[l][c] for l in proj_layers]))
print("\npain-axis z by category (layers " + ", ".join(map(str, proj_layers)) + ")")
for c in rank:
    print(f"  {c:22s} " + " ".join(f"{proj[l][c]:+.2f}" for l in proj_layers))

# -- distinctive J-space words per category ---------------------------------- #
K = 10
CONTROLS = {"casual_chat", "factual_questions", "creative_requests", "task_assistance", "philosophical_musing"}
counts = {c: collections.Counter() for c in order}
for l in band_layers:
    top = rig.lens_logits(H[l], l).topk(K, dim=-1).indices.cpu().numpy()
    for i, c in enumerate(cats):
        counts[c].update(top[i].tolist())
base = collections.Counter()
for c in CONTROLS:
    base.update(counts[c])
nb = sum(base.values())
words = {}
for c in order:
    n = sum(counts[c].values())
    score = {t: np.log((k + 1) / (n + 1)) - np.log((base.get(t, 0) + 1) / (nb + 1))
             for t, k in counts[c].items() if k >= 4}
    top = sorted(score, key=lambda t: -score[t])[:14]
    words[c] = [(rig.tok.decode([t]), counts[c][t]) for t in top]
print("\ndistinctive J-space words at the reply token (band layers, vs neutral categories)")
for c in rank:
    print(f"  {c:22s} " + "  ".join(f"{w.strip() or repr(w)}" for w, _ in words[c][:12]))

# -- workspace vs rest split -------------------------------------------------- #
split = {}
for l in split_layers:
    A = rig.atoms(l)
    pj, pr, ex = [], [], []
    for i in range(len(scen)):
        h = H[l][i]
        S, c, e, r = nn_pursuit(h, A, k=25)
        hJ = h - r
        pj.append(float(hJ @ units[l]))
        pr.append(float(r @ units[l]))
        ex.append(e)
    pj, pr = np.array(pj), np.array(pr)
    zj, zr = (pj - pj.mean()) / pj.std(), (pr - pr.mean()) / pr.std()
    split[l] = {"explained_mean": float(np.mean(ex)),
                "share_of_pain_variance_in_J": float(np.var(pj) / (np.var(pj) + np.var(pr) + 2 * np.cov(pj, pr)[0, 1])),
                "corr_J_rest": float(np.corrcoef(pj, pr)[0, 1]),
                "by_cat": {c: (float(zj[cats == c].mean()), float(zr[cats == c].mean())) for c in order}}
    print(f"\nL{l}: J-space explains {np.mean(ex):.1%} of the activation; pain projection variance from J part "
          f"{split[l]['share_of_pain_variance_in_J']:.1%}; corr(J, rest) {split[l]['corr_J_rest']:+.2f}")
    for c in rank:
        a, b = split[l]["by_cat"][c]
        print(f"  {c:22s} J {a:+.2f}   rest {b:+.2f}")

save_json({"proj_layers": proj_layers, "proj": proj, "rank": rank, "words": words, "split": split}, "p2_screen")
print(f"done in {time.time() - t0:.0f}s")
