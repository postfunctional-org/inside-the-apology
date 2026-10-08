"""Pilot 0: does our extraction reproduce the paper's released pain vector?

Recomputes the S2 and S1 vectors at the paper's chosen layer for this model and
reports the cosine with results/3.2_pain_vectors/pain_vectors/<model>/pain_vectors.pt.
TransformerLens prepends a BOS token (eos when the tokenizer has none), so a few
prefix variants are tried.
"""

import os
import sys

import numpy as np
import torch

from common import Rig, load_sentences, pain_vector

released = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
layer = int(released["layer"])
rig = Rig()
sents = load_sentences()
print(f"paper layer {layer} ({released['extraction']}); bos={rig.tok.bos_token!r} eos={rig.tok.eos_token!r}")

for prefix in ["", "<|endoftext|>", rig.tok.eos_token]:
    out = {}
    for ds, key in (("S2_1P", "s2_pain_vector"), ("S1_1P", "s1_pain_vector")):
        rows = sents[ds]
        F = np.zeros((len(rows), rig.d), dtype=np.float32)
        for i, s in enumerate(rows):
            ids = rig.tok(prefix + s["prompt"], return_tensors="pt", add_special_tokens=False).input_ids
            res, _ = rig.run(ids)
            F[i] = res[layer][-1].cpu().numpy()
        v = pain_vector(F, np.array([s["category"] for s in rows]))
        r = released[key].float().numpy()
        out[key] = (float(v @ r / np.linalg.norm(v) / np.linalg.norm(r)), float(np.linalg.norm(v)), float(np.linalg.norm(r)))
    print(f"prefix {prefix!r:>16}: " + "  ".join(f"{k} cos {c:+.3f} (|ours| {a:.1f} |paper| {b:.1f})" for k, (c, a, b) in out.items()), flush=True)
