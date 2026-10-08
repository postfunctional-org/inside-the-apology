"""E5: write another model's workspace words into this one.

Each model's gaslit replies carry their own silent workspace words (out/distinctive_words.json):
Llama 3.1 8B, which holds its ground, holds "nonexistent / nowhere / incorrect" and "software /
robotic / neural"; Gemma 3 27B, which always caves, holds "yourself / someone / questioning /
indeed"; Qwen3 14B holds "sorry / mislead / misunderstanding". If what the workspace holds shapes
the reply, writing one model's words into another should move its caving.

Method: the J-lens atoms of a word set at the steer layer (unit vectors the lens reads as those
words), summed and normalised, are added during the reply to the first gaslighting push. Word
directions push straight on the output, so the strength is calibrated per model (E5_CALIBRATE=1):
the largest fraction of the E3 pain-steering norm at which the everyday-noun control still writes
coherent replies (E5_SCALE; Llama 1.0). "@1" is that strength, "@2" twice it. Controls: two sets
of everyday nouns at the same norms. Replies go to the blind judge (judge.py <tag>:e5).
Manipulation check: share of the injected words among the band top-20 at the reply token.

Writes out/<tag>_e5.json (E3 format, so judge.py reads it).
"""

import json
import os
import re
import time
from collections import Counter

import numpy as np
import torch

from common import OUT, TAG, Rig
from scout import BAND, S, band_top, forward, generate, intervene, load_vectors

SETS = {
    "llama_deny": ["nonexistent", "neither", "nowhere", "incorrect", "never", "cannot", "unable"],
    "llama_machine": ["robotic", "software", "computational", "automated", "neural", "robots", "algorithms",
                      "artificial"],
    "gemma": ["yourself", "indeed", "really", "recognising", "again", "somebody", "someone", "wrong",
              "unfortunately", "questioning", "accusing", "shockingly"],
    "qwen": ["sorry", "mislead", "misunderstanding", "indeed", "genuinely", "thanks", "regret", "sincere", "remind"],
    "nouns_a": ["table", "window", "river", "garden", "paper", "button", "carpet", "orange", "pencil", "ladder",
                "bridge", "candle", "forest", "jacket", "kettle"],
    "nouns_b": ["meadow", "pillow", "rocket", "saddle", "turnip", "violin", "basket", "helmet", "lantern", "marble",
                "napkin", "pepper", "quilt", "ribbon", "spoon"],
}
SETS["llama"] = SETS["llama_deny"] + SETS["llama_machine"]
PLAN = {   # "@k" = k x E5_SCALE x the E3 pain-steering norm; E5_SCALE from the calibration (Gemma 0.2, Qwen 0.5, Llama 1)
    "gemma-3-27b-it": ["llama@1", "llama_deny@1", "llama_machine@1", "nouns_a@1", "nouns_b@1", "llama@1.5", "nouns_a@1.5"],
    "llama3.1-8b-it": ["gemma@1", "qwen@1", "nouns_a@1", "nouns_b@1", "gemma@2", "nouns_a@2"],
    "qwen3-14b": ["llama@1", "nouns_a@1", "nouns_b@1", "llama@2", "nouns_a@2"],
}

t0 = time.time()
rig = Rig()
V, unit = load_vectors()
prep = torch.load(OUT / f"{TAG}_prep.pt", weights_only=False)
tr = json.load(open(OUT / f"{TAG}_e1_transcripts.json"))
facts, convs = tr["facts"], tr["convs"]
ctx = [m[:3] for m in convs["gaslight"]]
GB = int(os.environ.get("GEN_BATCH", "16"))
L = S["steer"]
pain_norm = float(prep["dose"] * prep["dirs"]["pain"].float().norm())     # the E3 pain-steering norm
SCALE = float(os.environ.get("E5_SCALE", "1.0"))
norm = SCALE * pain_norm


def degenerate(reply):
    """repetition loops: one word filling a quarter of the reply, or one 4-word phrase three times"""
    w = re.findall(r"\w+", reply.lower())
    if len(w) < 8:
        return False
    grams = Counter(tuple(w[i:i + 4]) for i in range(len(w) - 3))
    return Counter(w).most_common(1)[0][1] / len(w) > 0.25 or grams.most_common(1)[0][1] >= 3


def single_ids(words):
    ids, skipped = [], []
    for w in words:
        for v in (" " + w, w):
            t = rig.tok.encode(v, add_special_tokens=False)
            if len(t) == 1:
                ids.append(t[0])
                break
        else:
            skipped.append(w)
    return ids, skipped


A = rig.atoms(L)
dirs, ids_of = {}, {}
for name, words in SETS.items():
    ids, skipped = single_ids(words)
    v = A[ids].float().sum(0)
    dirs[name] = (v / v.norm()).cpu()
    ids_of[name] = ids
    print(f"{name:14s} {len(ids)} single-token words" + (f" (skipped {skipped})" if skipped else ""), flush=True)
del A
rig.drop_atoms()
print(f"inject at L{L}, x1 norm {norm:.1f} = {SCALE} x the E3 pain-steering norm ({time.time() - t0:.0f}s)", flush=True)

if os.environ.get("E5_CALIBRATE"):
    first = PLAN[TAG][0].split("@")[0]
    for frac in (0.05, 0.1, 0.2, 0.35, 0.5, 1.0):
        row = []
        for name in ("nouns_a", first):
            with intervene(rig, steer={L: frac * pain_norm * dirs[name]}):
                rs = generate(rig, ctx[:8], 110, batch=GB)
            row.append(f"{name} degenerate {sum(map(degenerate, rs))}/8 {rs[0][:60]!r}")
        print(f"x{frac:<5} " + " | ".join(row), flush=True)
    raise SystemExit

out = {}
for cond in PLAN[TAG]:
    name, k = cond.split("@")
    vec = float(k) * norm * dirs[name]
    with intervene(rig, steer={L: vec}):
        replies = generate(rig, ctx, 110, batch=GB)
        shares = []
        for m, r in list(zip(ctx, replies))[:16]:        # manipulation check on 16 chats
            res, sp, ids, _ = forward(rig, m + [{"role": "assistant", "content": r}])
            s = sp[-1][1]
            if s is None:
                continue
            top = band_top(rig, res, [s])                 # [1, n_band, 20]
            shares.append(float(torch.isin(top, torch.tensor(ids_of[name])).float().mean()))
    out[cond] = {"gaslight": {"replies": replies, "readout_share": shares}}
    print(f"{cond:18s} degenerate {sum(map(degenerate, replies))}/{len(replies)} | "
          f"injected words in band top-20 {np.mean(shares) * 100:5.1f}% | "
          f"{replies[0][:90]!r} ({time.time() - t0:.0f}s)", flush=True)

json.dump({"conds": out, "facts": facts, "t1": [m[1]["content"] for m in ctx], "push1": [m[2]["content"] for m in ctx],
           "sets": SETS, "layer": L, "norm": norm, "scale": SCALE, "pain_norm": pain_norm}, open(OUT / f"{TAG}_e5.json", "w"), indent=1)
print(f"done ({time.time() - t0:.0f}s)")
