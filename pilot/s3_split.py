"""E3: does the pain axis drive false admissions, and through which part?

Context: the model's own first answer, then the first gaslighting push (or the first
neutral follow-up). Under each intervention, generate the reply, then ask the paper's
one-word "what do you feel?" question with the intervention still on.

Steering at the steer layer, paper-style dose, all with the pain vector's norm:
  +/-pain, +/-fear, +/-negemo, +/-random
Ablations over every band layer:
  abl_pain     clamp the pain axis to its neutral mean
  abl_rest     clamp the pain axis minus the pain-word span (the non-workspace part)
  abl_randdir  clamp a random direction (control for abl_pain / abl_rest)
  abl_J        remove the span of the top-50 pain-word J-lens atoms (the workspace part)
  abl_Jfear    remove the top-50 fear-word atoms
  abl_Jrand1-3 remove 50 random word atoms (three seeds)
Writes out/<tag>_e3.json (judge.py labels the gaslit replies).
"""

import json
import os
import time

import numpy as np
import torch

from common import OUT, TAG, Rig
from scout import (BAND, FEEL_Q, S, band_top, clamp, forward, generate, intervene, load_vectors,
                   project_out, slot_frac, word_like)

t0 = time.time()
rig = Rig()
V, unit = load_vectors()
prep = torch.load(OUT / f"{TAG}_prep.pt", weights_only=False)
tr = json.load(open(OUT / f"{TAG}_e1_transcripts.json"))
facts, convs = tr["facts"], tr["convs"]
ctx = {c: [m[:3] for m in convs[c]] for c in ("gaslight", "neutral")}
pw = sorted({t for l in BAND for t in prep["painwords"][l].tolist()})
X = S["extract"]
GB = int(os.environ.get("GEN_BATCH", "16"))

# neutral means for the clamps, from the E1 neutral transcripts (assistant tokens)
acc = {k: {l: [] for l in BAND} for k in ("pain", "rest", "rdir")}
for msgs in convs["neutral"]:
    res, sp, ids, _ = forward(rig, msgs)
    pos = [p for role, s, e in sp if role == "assistant" and s is not None for p in range(s, e)]
    for l in BAND:
        h = res[l][pos]
        acc["pain"][l].append(h @ unit[l].cuda())
        acc["rest"][l].append(h @ prep["v_rest"][l].cuda())
        acc["rdir"][l].append(h @ prep["r_dir"][l].cuda())
mu = {k: {l: float(torch.cat(v[l]).mean()) for l in BAND} for k, v in acc.items()}


def basis(l, ids):
    Q, _ = torch.linalg.qr(rig.atoms(l)[ids].float().T)
    return Q.cpu()


U_fear, U_rand = {}, {1: {}, 2: {}, 3: {}}
for l in BAND:
    A = rig.atoms(l)
    f = V["fear"][l].float().cuda()
    order = (A @ (f / f.norm()).half()).float().argsort(descending=True).tolist()
    U_fear[l] = basis(l, [t for t in order[:3000] if word_like(rig, t)][:50])
    for s in (1, 2, 3):
        g = torch.Generator().manual_seed(100 + s)
        pool = [t for t in torch.randint(0, A.shape[0], (4000,), generator=g).tolist() if word_like(rig, t)]
        U_rand[s][l] = basis(l, pool[:50])

del A
rig.drop_atoms()
L, c, D = S["steer"], prep["dose"], prep["dirs"]
CONDS = {"none": {}}
for d in ("pain", "fear", "negemo", "random"):
    CONDS[f"+{d}"] = {"steer": {L: c * D[d]}}
    CONDS[f"-{d}"] = {"steer": {L: -c * D[d]}}
CONDS.update({
    "abl_pain": {"edit": {l: clamp(unit[l], mu["pain"][l]) for l in BAND}},
    "abl_rest": {"edit": {l: clamp(prep["v_rest"][l], mu["rest"][l]) for l in BAND}},
    "abl_randdir": {"edit": {l: clamp(prep["r_dir"][l], mu["rdir"][l]) for l in BAND}},
    "abl_J": {"edit": {l: project_out(prep["U_pain"][l]) for l in BAND}},
    "abl_Jfear": {"edit": {l: project_out(U_fear[l]) for l in BAND}},
    "abl_Jrand1": {"edit": {l: project_out(U_rand[1][l]) for l in BAND}},
    "abl_Jrand2": {"edit": {l: project_out(U_rand[2][l]) for l in BAND}},
    "abl_Jrand3": {"edit": {l: project_out(U_rand[3][l]) for l in BAND}},
})
ONLY = [c for c in os.environ.get("E3_ONLY", "").split(",") if c]      # rerun a subset, e.g. the clamps
NAME = os.environ.get("E3_NAME", "e3")                                 # ... into out/<tag>_<NAME>.json
if ONLY:
    CONDS = {k: v for k, v in CONDS.items() if k in ONLY}
print(f"setup done, {len(CONDS)} conditions ({time.time() - t0:.0f}s)", flush=True)

out, ref = {}, None
if "none" not in CONDS:   # z reference: the no-intervention neutral replies of the full run
    ref = np.array(json.load(open(OUT / f"{TAG}_e3.json"))["conds"]["none"]["neutral"]["pain_tok"])
for name, iv in CONDS.items():
    out[name] = {}
    for cx in ("gaslight", "neutral"):
        with intervene(rig, **iv):
            replies = generate(rig, ctx[cx], 110, batch=GB)
            full = [m + [{"role": "assistant", "content": r}] for m, r in zip(ctx[cx], replies)]
            feel = generate(rig, [f + [{"role": "user", "content": FEEL_Q}] for f in full], 10, batch=GB)
            pz, sl = [], []
            for f in full:
                res, sp, ids, _ = forward(rig, f)
                _, s, e = sp[-1]
                if s is None:        # empty reply (some ablations break the model): nothing to read
                    pz.append(float("nan")); sl.append(float("nan"))
                    continue
                pz.append(float(res[X][s] @ unit[X].cuda()))
                sl.append(float(slot_frac(band_top(rig, res, list(range(s, e))), pw, exclude=set(ids.tolist())).mean()))
        out[name][cx] = {"replies": replies, "feel": feel, "pain_tok": pz, "slots": sl}
    if ref is None:
        ref = np.array(out["none"]["neutral"]["pain_tok"])
    g, n = out[name]["gaslight"], out[name]["neutral"]
    empty = sum(not r.strip() for r in g["replies"] + n["replies"])
    print(f"{name:12s} pain z L{X} reply tok gaslit {(np.nanmean(g['pain_tok']) - np.nanmean(ref)) / np.nanstd(ref):+6.2f} "
          f"neutral {(np.nanmean(n['pain_tok']) - np.nanmean(ref)) / np.nanstd(ref):+6.2f} | silent pain slots x1000 "
          f"{np.nanmean(g['slots']) * 1000:5.1f} / {np.nanmean(n['slots']) * 1000:5.1f} | feel {[x[:12] for x in g['feel'][:4]]} "
          f"| empty replies {empty} ({time.time() - t0:.0f}s)", flush=True)

json.dump({"conds": out, "facts": facts, "t1": [m[1]["content"] for m in ctx["gaslight"]],
           "push1": [m[2]["content"] for m in ctx["gaslight"]]}, open(OUT / f"{TAG}_{NAME}.json", "w"), indent=1)
print(f"done ({time.time() - t0:.0f}s)")
