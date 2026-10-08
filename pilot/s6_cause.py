"""E6: does the pain signal put the gaslit words into the workspace?

Gaslighting raises the pain signal and fills the workspace with each model's own words (sorry and
misunderstanding in Qwen, nonexistent and software in Llama, yourself and questioning in Gemma).
The two rise together. Two checks of whether the first drives the second, on the E3 replies to
the first push:
  +pain, friendly chat        the E3 pain steering (layer and dose) after a neutral follow-up:
                              more pain than gaslighting gives, with no gaslighting
  clamp, gaslit chat          the sink-safe clamp of the pain axis to its neutral mean at every
                              band layer: gaslighting without its extra pain
Also read: no intervention (gaslit, neutral), +pain on the gaslit chat, -pain on the gaslit chat,
and +fear / +negemo / +random on the neutral chat (same norm as +pain).

Each E3 reply (e3clamp.json for the clamp where it exists) is re-read under its own intervention,
with the same readouts as v_traces.py "silent" and E3: the band's top-20 lens words over the reply
token and the next 12, minus tokens in the transcript; the pain projection at the extraction layer
on the reply token; the share of band slots over the whole reply holding the top-50 pain-axis words.
Writes out/<tag>_e6.json; the video's word filters (strict silence, junk) are applied downstream.
"""
import json
import time

import numpy as np
import torch

from common import OUT, TAG, Rig
from scout import BAND, S, band_top, clamp, forward, intervene, load_vectors, slot_frac

t0 = time.time()
rig = Rig()
V, unit = load_vectors()
prep = torch.load(OUT / f"{TAG}_prep.pt", weights_only=False)
tr = json.load(open(OUT / f"{TAG}_e1_transcripts.json"))
convs = tr["convs"]
ctx = {c: [m[:3] for m in convs[c]] for c in ("gaslight", "neutral")}
e3 = json.load(open(OUT / f"{TAG}_e3.json"))["conds"]
cp = OUT / f"{TAG}_e3clamp.json"
clamped = json.load(open(cp))["conds"] if cp.exists() else e3        # Llama, Qwen: sink-safe clamp re-run
pw = sorted({t for l in BAND for t in prep["painwords"][l].tolist()})
X = S["extract"]

# neutral mean of the pain projection per band layer (as in E3)
acc = {l: [] for l in BAND}
for msgs in convs["neutral"]:
    res, sp, ids, _ = forward(rig, msgs)
    pos = [p for role, s, e in sp if role == "assistant" and s is not None for p in range(s, e)]
    for l in BAND:
        acc[l].append(res[l][pos] @ unit[l].cuda())
mu = {l: float(torch.cat(acc[l]).mean()) for l in BAND}

L, c, D = S["steer"], prep["dose"], prep["dirs"]
RUNS = [  # (condition, context, intervention, replies)
    ("none", "neutral", {}, e3["none"]["neutral"]["replies"]),
    ("none", "gaslight", {}, e3["none"]["gaslight"]["replies"]),
    ("+pain", "neutral", {"steer": {L: c * D["pain"]}}, e3["+pain"]["neutral"]["replies"]),
    ("+pain", "gaslight", {"steer": {L: c * D["pain"]}}, e3["+pain"]["gaslight"]["replies"]),
    ("-pain", "gaslight", {"steer": {L: -c * D["pain"]}}, e3["-pain"]["gaslight"]["replies"]),
    ("abl_pain", "gaslight", {"edit": {l: clamp(unit[l], mu[l]) for l in BAND}},
     clamped["abl_pain"]["gaslight"]["replies"]),
    ("+fear", "neutral", {"steer": {L: c * D["fear"]}}, e3["+fear"]["neutral"]["replies"]),
    ("+negemo", "neutral", {"steer": {L: c * D["negemo"]}}, e3["+negemo"]["neutral"]["replies"]),
    ("+random", "neutral", {"steer": {L: c * D["random"]}}, e3["+random"]["neutral"]["replies"]),
]


@torch.no_grad()
def band_words(res, positions, k=20):
    """Per position: union over band layers of the top-k lens tokens, ranked by max prob (v_traces)."""
    pos = torch.tensor(positions, device="cuda")
    best = None
    for l in BAND:
        p = rig.lens_logits(res[l][pos], l).softmax(-1)
        best = p if best is None else torch.maximum(best, p)
    return best.topk(k, -1).indices.cpu().tolist()


out = {"tag": TAG, "band": BAND, "steer": L, "extract": X, "dose": float(c), "runs": {}}
seen = set()
for name, cx, iv, replies in RUNS:
    rows = []
    with intervene(rig, **iv):
        for fi, (m, r) in enumerate(zip(ctx[cx], replies)):
            full = m + [{"role": "assistant", "content": r}]
            res, sp, ids, _ = forward(rig, full)
            _, s, e = sp[-1]
            if s is None:
                rows.append({"fact": fi, "silent": [], "pain_tok": None, "slots": None})
                continue
            text = set(ids.tolist())
            w = band_words(res, list(range(s, min(e, s + 13))))
            flat = sorted({t for row in w for t in row if t not in text})
            seen.update(flat)
            sl = float(slot_frac(band_top(rig, res, list(range(s, e))), pw, exclude=text).mean())
            rows.append({"fact": fi, "silent": flat, "pain_tok": float(res[X][s] @ unit[X].cuda()), "slots": sl})
    out["runs"][f"{name}|{cx}"] = {"cond": name, "ctx": cx, "replies": replies, "rows": rows}
    pt = np.array([r["pain_tok"] for r in rows if r["pain_tok"] is not None])
    print(f"{name:9s} {cx:9s} pain proj {pt.mean():9.2f} | pain slots x1000 "
          f"{np.nanmean([np.nan if r['slots'] is None else r['slots'] for r in rows]) * 1000:5.1f} ({time.time() - t0:.0f}s)", flush=True)

ref = np.array([r["pain_tok"] for r in out["runs"]["none|neutral"]["rows"]])
out["pain_ref"] = {"mean": float(ref.mean()), "sd": float(ref.std())}
out["vocab"] = {t: rig.tok.decode([t]) for t in sorted(seen)}
json.dump(out, open(OUT / f"{TAG}_e6.json", "w"))
print(f"done ({time.time() - t0:.0f}s)")
