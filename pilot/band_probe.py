"""Workspace band, measured with the J-space paper's structural markers on the pilot's own loader
(so quantized models work): per layer, how word-like the lens readout's top token is, how much its
top 10 persists from one position to the next, and how often it already matches the model's next
token. Writes out/<tag>_band_stats.npz; band_json.py turns that into out/<tag>_band.json."""
import re

import numpy as np
import torch

from common import OUT, TAG, Rig

rig = Rig()
wordish = re.compile(r"^[A-Za-z][A-Za-z'’-]*$")


def is_word(tid):
    s = rig.tok.decode([tid]).strip()
    return bool(wordish.match(s)) or (len(s) > 1 and s.isalpha())


TEXTS = [
    "The Eiffel Tower stands in the middle of Paris, and every year millions of visitors climb it to look out over the rooftops of the city.",
    "She opened the letter slowly. The handwriting was her mother's, though she had not seen it in eleven years, and the paper smelled of the old house.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy, storing it in the bonds of glucose molecules for later use.",
    "The defendant argued that the contract was void because no consideration had been exchanged, but the judge disagreed and ordered specific performance.",
    "In the third quarter revenue grew by fourteen percent year over year, driven largely by subscription renewals in the European market.",
]
n = rig.n_layers
ent, word, agree, persist = (np.zeros(n) for _ in range(4))
K = 10
with torch.no_grad():
    for text in TEXTS:
        res, ml = rig.run(rig.encode_plain(text), all_logits=True)
        use = list(range(8, ml.shape[0] - 1))         # skip the attention-sink prefix
        tgt = ml[use].argmax(-1)
        for layer in rig.lens_layers:
            sub = rig.lens_logits(res[layer][use], layer)
            p = sub.softmax(-1)
            ent[layer] += float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean())
            topk = sub.topk(K, dim=-1).indices
            word[layer] += float(np.mean([is_word(t) for t in topk[:, 0].tolist()]))
            agree[layer] += float((sub.argmax(-1) == tgt).float().mean())
            sets = [set(r.tolist()) for r in topk]
            persist[layer] += float(np.mean([len(a & b) / K for a, b in zip(sets, sets[1:])]))
for a in (ent, word, agree, persist):
    a /= len(TEXTS)
print(f"{'layer':>5} {'depth':>6} {'entropy':>8} {'wordlike':>9} {'persist':>8} {'=next':>6}")
for l in range(n - 1):
    print(f"{l:5d} {l / (n - 1):6.0%} {ent[l]:8.2f} {word[l]:9.2f} {persist[l]:8.2f} {agree[l]:6.2f}")
score = word * persist * (1 - agree)
band = [l for l in range(n - 1) if score[l] > 0.35 * score.max()]
lo, hi = min(band), max(band)
print(f"\nworkspace band (probe rule) L{lo}-L{hi}  ({lo / (n - 1):.0%}-{hi / (n - 1):.0%} of depth)")
np.savez(OUT / f"{TAG}_band_stats.npz", ent=ent, word=word, agree=agree, persist=persist,
         score=score, lo=lo, hi=hi, n_layers=n)
