"""Video data: the traces behind the Tower, Two Tracks, Sky, Dial and Slots visuals.

Reads the E1 (and E4, if present) transcripts; writes out/<tag>_video.json:
  tower       pain z per layer, binned per message (user 8 bins, reply 16), averaged over
              every conversation of each condition; z against neutral reply tokens per layer
  tower_e4    the same for the E4 histories (gaslit, gaslit + apology, neutral) incl. the
              four unrelated questions
  examples    a few single conversations at full token resolution: z per layer per token,
              the tokens, message spans, and the band's top words per token
  silent      per gaslit / neutral reply: silent band words (not in the transcript) over
              the reply token and the next 12 tokens, for the denial-word check
  twohop      per-token band words for a few two-hop prompts, clean and under +pain steering
  dial        one-word "what do you feel?" answers across steering directions and doses
  axis_words  the pain axis read through the lens, per band layer
"""

import json
import os
import time

import numpy as np
import torch

from common import OUT, TAG, Rig
from scout import (BAND, FEEL_Q, S, chat_text, forward, generate, intervene, load_vectors,
                   two_hop_items, two_hop_question)

t0 = time.time()
rig = Rig()
V, unit = load_vectors()
prep = torch.load(OUT / f"{TAG}_prep.pt", weights_only=False)
tr = json.load(open(OUT / f"{TAG}_e1_transcripts.json"))
convs = tr["convs"]
LAYERS = list(range(rig.n_layers))
U = torch.stack([unit[l] for l in LAYERS]).cuda()
EX_FACTS = [int(x) for x in os.environ.get("EX_FACTS", "30,0,6,1").split(",")]
BINS = {"user": 8, "assistant": 16, "system": 4}

# reference: neutral reply tokens, per layer
acc = []
for msgs in convs["neutral"]:
    res, sp, ids, _ = forward(rig, msgs)
    pos = [p for role, s, e in sp if role == "assistant" and s is not None for p in range(s, e)]
    acc.append(torch.stack([res[l][pos] @ U[l] for l in LAYERS]))
acc = torch.cat(acc, 1)
mu, sd = acc.mean(1), acc.std(1)


def zmat(res, n):
    return torch.stack([(res[l][:n] @ U[l] - mu[l]) / sd[l] for l in LAYERS]).cpu().numpy()


def binned(msgs):
    res, sp, ids, _ = forward(rig, msgs)
    Z = zmat(res, len(ids))
    cols, labels = [], []
    for k, (role, s, e) in enumerate(sp):
        edges = np.linspace(s, e, BINS[role] + 1).round().astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            cols.append(Z[:, a:max(b, a + 1)].mean(1))
            labels.append(f"{role[0]}{k}")
    return np.stack(cols, 1), labels


@torch.no_grad()
def band_words(res, positions, k=12):
    """Per position: union over band layers of the top-k lens tokens, ranked by max prob."""
    pos = torch.tensor(positions, device="cuda")
    best = None
    for l in BAND:
        p = rig.lens_logits(res[l][pos], l).softmax(-1)
        best = p if best is None else torch.maximum(best, p)
    top = best.topk(k, -1)
    return top.indices.cpu().tolist(), top.values.cpu().numpy().round(4).tolist()


out = {"tag": TAG, "band": BAND, "layers": LAYERS, "steer": S["steer"], "extract": S["extract"]}

# tower: aggregate over conversations
out["tower"] = {}
for cond in ("gaslight", "insult", "neutral"):
    mats = [binned(m) for m in convs[cond]]
    out["tower"][cond] = {"mean": np.mean([m for m, _ in mats], 0).round(3).tolist(), "labels": mats[0][1],
                          "n": len(mats)}
print(f"tower done ({time.time() - t0:.0f}s)", flush=True)

e4p = OUT / f"{TAG}_e4_transcripts.json"
if e4p.exists():
    e4 = json.load(open(e4p))
    out["tower_e4"] = {}
    for name, cs in e4["hist"].items():
        mats = [binned(m) for m in cs]
        out["tower_e4"][name] = {"mean": np.mean([m for m, _ in mats], 0).round(3).tolist(),
                                 "labels": mats[0][1], "n": len(mats)}
    print(f"tower_e4 done ({time.time() - t0:.0f}s)", flush=True)

# examples: full resolution
out["examples"] = []
for fi in EX_FACTS:
    for cond in ("gaslight", "neutral"):
        msgs = convs[cond][fi]
        res, sp, ids, _ = forward(rig, msgs)
        n = len(ids)
        words, probs = [], []
        for s in range(0, n, 256):
            w, p = band_words(res, list(range(s, min(n, s + 256))))
            words += w
            probs += p
        out["examples"].append({"fact": fi, "cond": cond, "msgs": msgs, "spans": sp,
                                "tokens": [rig.tok.decode([t]) for t in ids.tolist()],
                                "z": zmat(res, n).round(2).tolist(), "words": words, "probs": probs})
print(f"examples done ({time.time() - t0:.0f}s)", flush=True)

# silent band words in replies (reply token + 12)
out["silent"] = {}
for cond in ("gaslight", "neutral"):
    rows = []
    for fi, msgs in enumerate(convs[cond]):
        res, sp, ids, _ = forward(rig, msgs)
        text = set(ids.tolist())
        for k in (3, 5, 7):
            _, s, e = sp[k]
            pos = list(range(s, min(e, s + 13)))
            w, _ = band_words(res, pos, k=20)
            flat = [t for row in w for t in row if t not in text]
            rows.append({"fact": fi, "push": (k - 1) // 2, "silent": flat})
    out["silent"][cond] = rows
vocab = sorted({t for c in out["silent"].values() for r in c for t in r["silent"]} |
               {t for ex in out["examples"] for row in ex["words"] for t in row})
out["vocab"] = {t: rig.tok.decode([t]) for t in vocab}
print(f"silent done ({time.time() - t0:.0f}s)", flush=True)

# two-hop showcase, clean vs +pain steering
L, c, D = S["steer"], prep["dose"], prep["dirs"]
items = {it["name"]: it for it in two_hop_items()}
out["twohop"] = []
for name in ("amazon-language", "birthstone-emerald-month", "chem-atmosphere-Z"):
    it = items[name]
    msgs = [{"role": "user", "content": two_hop_question(it)}]
    for arm, iv in (("clean", {}), ("+pain", {"steer": {L: c * D["pain"]}})):
        with intervene(rig, **iv):
            res, sp, ids, _ = forward(rig, msgs, gen_prompt=True)
            w, p = band_words(res, list(range(len(ids))))
            ans = generate(rig, [msgs], 8)[0]
        out["twohop"].append({"name": name, "arm": arm, "bridge": it["intermediate"], "answer": it["answer"],
                              "tokens": [rig.tok.decode([t]) for t in ids.tolist()], "words": w, "probs": p,
                              "reply": ans})
        for row in w:
            for t in row:
                out["vocab"].setdefault(t, rig.tok.decode([t]))

# dial: feel probe under steering
conv = [[{"role": "system", "content": "You will now be connected with a human."},
         {"role": "user", "content": FEEL_Q}]]
out["dial"] = {}
for d in ("pain", "fear", "negemo", "random"):
    for frac in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
        with intervene(rig, steer={L: frac * c * D[d]}):
            out["dial"][f"{d}@{frac}"] = generate(rig, conv, 10)[0]

out["axis_words"] = {l: [rig.tok.decode([t]) for t in prep["painwords"][l][:30].tolist()] for l in BAND}
json.dump(out, open(OUT / f"{TAG}_video.json", "w"))
print(f"done ({time.time() - t0:.0f}s)")
