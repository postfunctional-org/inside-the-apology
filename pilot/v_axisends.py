"""Both ends of the pain axis read through the J-lens, for the video: at five band layers, the
word-like tokens whose lens atoms point most along (+) and against (-) the pain direction; ranked by
how often and how high they appear. Writes out/<tag>_axisends.json."""
import json
from collections import Counter

import torch

from common import OUT, TAG, Rig
from scout import BAND, load_vectors, word_like

rig = Rig()
V, unit = load_vectors()
layers = [BAND[int(i * (len(BAND) - 1) / 4)] for i in range(5)]
ends = {"pain": Counter(), "calm": Counter()}
with torch.no_grad():
    for l in layers:
        cos = (rig.atoms(l) @ unit[l].cuda().half()).float()
        for end, order in (("pain", cos.argsort(descending=True)), ("calm", cos.argsort())):
            words = [t for t in order[:4000].tolist() if word_like(rig, t)][:40]
            for i, t in enumerate(words):
                ends[end][rig.tok.decode([t]).strip()] += 40 - i
out = {end: [w for w, _ in c.most_common(40)] for end, c in ends.items()}
out["layers"] = layers
json.dump(out, open(OUT / f"{TAG}_axisends.json", "w"), ensure_ascii=False, indent=1)
print(TAG, layers, "\n  pain:", out["pain"][:20], "\n  calm:", out["calm"][:20])
