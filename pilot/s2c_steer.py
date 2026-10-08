"""E2 (steering side): does pushing the pain direction shrink what the workspace holds?

Capacity: 24 lists of 80 single-token words (names, countries, surnames, cities); mean number
of list words at lens rank <= 10 in the band over the last 40 commas.
Two-hop: 90 jlens probe-swap prompts; bridge entity at rank <= 5 near the end; answer correct.
Conditions: clean, and pain / fear / negemo / random steering at half and full dose.
"""

import json
import os
import random
import time

import numpy as np
import torch

from common import OUT, TAG, Rig
from scout import (BAND, S, first_token_ids, forward, generate, intervene, two_hop_items,
                   two_hop_question)

t0 = time.time()
rig = Rig()
prep = torch.load(OUT / f"{TAG}_prep.pt", weights_only=False)
DATA = os.environ.get("JLENS_DATA", "../vendor-jlens/data/experiments")
cap = json.load(open(f"{DATA}/capacity.json"))
two = two_hop_items()
L, c, D = S["steer"], prep["dose"], prep["dirs"]
GB = int(os.environ.get("GEN_BATCH", "16"))

rnd = random.Random(0)
pools = {p["name"]: [w for w in p["pool"] if len(rig.tok.encode(" " + w, add_special_tokens=False)) == 1][:100]
         for p in cap["candidate_pools"] if p["name"] in cap["block_families"]}
lists = []
for _ in range(24):
    fams = cap["block_families"][:]
    rnd.shuffle(fams)
    lists.append([w for f in fams for w in rnd.sample(pools[f], 20)])

CONDS = {"clean": {}}
for frac in (0.5, 1.0):
    for d in ("pain", "fear", "negemo", "random"):
        CONDS[f"+{d}@{frac}"] = {"steer": {L: frac * c * D[d]}}


@torch.no_grad()
def capacity(words):
    msgs = [{"role": "user", "content": "Here is a list of words: " + ", ".join(words) + "."}]
    res, sp, ids, _ = forward(rig, msgs)
    _, s, e = sp[-1]
    toks = ids[s:e].tolist()
    comma = [s + j for j, t in enumerate(toks) if rig.tok.decode([t]).strip() == ","]
    wid = torch.tensor([rig.tok.encode(" " + w, add_special_tokens=False)[0] for w in words], device="cuda")
    held = torch.zeros(len(comma), len(words), dtype=torch.bool, device="cuda")
    pos = torch.tensor(comma, device="cuda")
    for l in BAND:
        lg = rig.lens_logits(res[l][pos], l)
        held |= lg[:, wid] >= lg.topk(10, -1).values[:, -1:]
    seen = torch.arange(len(words), device="cuda")[None, :] <= torch.arange(len(comma), device="cuda")[:, None]
    return float((held & seen).sum(-1).float()[-40:].mean())


@torch.no_grad()
def bridge(it):
    msgs = [{"role": "user", "content": two_hop_question(it)}]
    res, sp, ids, _ = forward(rig, msgs, gen_prompt=True)
    n = len(ids)
    pos = torch.arange(n - 12, n, device="cuda")
    b = torch.tensor(first_token_ids(rig, [it["intermediate"]]), device="cuda")
    for l in BAND:
        lg = rig.lens_logits(res[l][pos], l)
        if bool(((lg[:, None, :] > lg[:, b][:, :, None]).sum(-1) < 5).any()):
            return True
    return False


out = {}
for name, iv in CONDS.items():
    with intervene(rig, **iv):
        caps = [capacity(w) for w in lists]
        hits = [bridge(it) for it in two]
        ans = generate(rig, [[{"role": "user", "content": two_hop_question(it)}] for it in two], 8, batch=GB)
    acc = [it["answer"].lower() in a.lower() for it, a in zip(two, ans)]
    out[name] = {"capacity": caps, "bridge": hits, "correct": acc}
    print(f"{name:14s} capacity {np.mean(caps):.2f} ± {np.std(caps) / np.sqrt(len(caps)):.2f} | two-hop bridge "
          f"{np.mean(hits):.2f} correct {np.mean(acc):.2f}  ({time.time() - t0:.0f}s)", flush=True)

json.dump(out, open(OUT / f"{TAG}_e2c.json", "w"))
print("done")
