"""Pilot 5: silent two-hop reasoning as the planning probe.

jlens data/experiments/probe-swap.json: 90 prompts whose answer needs an
unstated bridge entity ("...the country where the Amazon River ends is" ->
Brazil -> Portuguese). Per item: is the greedy answer right, and does the
bridge reach J-lens rank <= 5 (and rank 1) in the band over the last prompt
tokens?
"""

import json
import os

import numpy as np

from common import Rig

band = [int(x) for x in os.environ.get("PILOT_BAND", "17,31").split(",")]
layers = list(range(band[0], band[1] + 1))
items = json.load(open(os.environ.get("JLENS_DATA", "../vendor-jlens/data/experiments") + "/probe-swap.json"))["items"]
rig = Rig()


def first_ids(word):
    return {rig.tok.encode(w, add_special_tokens=False)[0] for w in (word, " " + word)}


rows = []
for it in items:
    ids = rig.encode_plain(it["prompt"].rstrip())
    res, logits = rig.run(ids)
    correct = int(logits[-1].argmax()) in first_ids(it["answer"])
    bridge = first_ids(it["intermediate"])
    best = 10 ** 9
    for l in layers:
        lg = rig.lens_logits(res[l][-6:], l)  # last 6 prompt positions
        for b in bridge:
            best = min(best, int((lg > lg[:, b:b + 1]).sum(-1).min()))
    rows.append((it["name"], correct, best))

c = np.array([r[1] for r in rows])
b = np.array([r[2] for r in rows])
print(f"answer correct {c.mean():.0%}; bridge rank<=5 {np.mean(b <= 4):.0%}, rank 1 {np.mean(b == 0):.0%}; "
      f"among correct: rank<=5 {np.mean(b[c] <= 4):.0%}")
for name, ok, r in rows[:12]:
    print(f"  {name:28s} correct={ok!s:5s} bridge best rank {r + 1}")
