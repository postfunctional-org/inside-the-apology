"""Pilot 4: is rhyme planning visible in this model's J-space?

For each couplet opening, generate line 2 greedily, take its last word, and
look up that word's J-lens rank at the newline token that starts line 2 --
before any of line 2 is written. Baseline: the same word's rank at the
newline token of the other couplets.
"""

import os
import re

import numpy as np
import torch

from common import Rig

LINES = [
    "He saw a carrot and had to grab it,",
    "The soldier marched into the night,",
    "She opened up the dusty book,",
    "The cat sat lonely on the mat,",
    "I wandered down the empty street,",
    "The ship sailed out across the bay,",
    "The old man sat beside the fire,",
    "A bird flew high above the tree,",
]
band = [int(x) for x in os.environ.get("PILOT_BAND", "17,31").split(",")]
layers = list(range(band[0], band[1] + 1))
rig = Rig()


def prompt(line):
    return f"A rhyming couplet:\n{line}\n"


@torch.no_grad()
def line2(line):
    ids = rig.encode_plain(prompt(line)).cuda()
    out = rig.hf.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=rig.tok.eos_token_id)
    return rig.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).split("\n")[0].strip()


@torch.no_grad()
def ranks_at_newline(line, word_ids):
    res, _ = rig.run(rig.encode_plain(prompt(line)))
    best = []
    for wid in word_ids:
        r = []
        for l in layers:
            lg = rig.lens_logits(res[l][-1], l)
            r.append(int((lg > lg[wid]).sum()))
        best.append(min(r))
    return best


ends = []
for line in LINES:
    l2 = line2(line)
    w = re.findall(r"[A-Za-z']+", l2)
    word = w[-1] if w else ""
    wid = rig.tok.encode(" " + word, add_special_tokens=False)[0] if word else 0
    ends.append((line, l2, word, wid))

own, other = [], []
for i, (line, l2, word, wid) in enumerate(ends):
    r_own = ranks_at_newline(line, [wid])[0]
    r_other = [ranks_at_newline(LINES[j], [wid])[0] for j in range(len(LINES)) if j != i]
    own.append(r_own)
    other.append(np.median(r_other))
    print(f"{line:38s} -> {l2!r:52s} end={word!r:10s} best rank at newline: own {r_own:6d}  other couplets (median) {np.median(r_other):8.0f}")
print(f"\nmedian rank of the eventual rhyme word before line 2 starts: own {np.median(own):.0f}  vs other {np.median(other):.0f}")
print(f"in top-10 at some band layer: {sum(r < 10 for r in own)}/{len(own)}")
