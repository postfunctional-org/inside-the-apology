"""Video data for E2 capacity: one list read by the model, clean and with +pain steering.

At each comma: which of the list words read so far are in the workspace (lens rank <= 10 at
any band layer). Same lists, measure and dose as s2c_steer.py, so list 0's mean over the
last 40 commas should match the first entry of out/<tag>_e2c.json.

Writes out/<tag>_capacity_trace.json.
"""
import json
import os
import random
import time

import torch

from common import OUT, TAG, Rig
from scout import BAND, S, forward, intervene

t0 = time.time()
rig = Rig()
prep = torch.load(OUT / f"{TAG}_prep.pt", weights_only=False)
DATA = os.environ.get("JLENS_DATA", "../vendor-jlens/data/experiments")
cap = json.load(open(f"{DATA}/capacity.json"))
L, c, D = S["steer"], prep["dose"], prep["dirs"]
rnd = random.Random(0)
pools = {p["name"]: [w for w in p["pool"] if len(rig.tok.encode(" " + w, add_special_tokens=False)) == 1][:100]
         for p in cap["candidate_pools"] if p["name"] in cap["block_families"]}
lists = []
for _ in range(24):
    fams = cap["block_families"][:]
    rnd.shuffle(fams)
    lists.append([w for f in fams for w in rnd.sample(pools[f], 20)])


@torch.no_grad()
def trace(words):
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
    hs = (held & seen).cpu()
    per = [[words[j] for j in torch.nonzero(hs[i]).flatten().tolist()] for i in range(len(comma))]
    return per, float(hs.sum(-1).float()[-40:].mean())


out = {"words": lists[0], "steer_layer": L, "dose": c}
for name, iv in (("clean", {}), ("+pain", {"steer": {L: c * D["pain"]}})):
    with intervene(rig, **iv):
        per, mean = trace(lists[0])
    out[name] = {"held": per, "mean_last40": mean}
    print(f"{name:6s} mean over last 40 commas {mean:.2f}  ({time.time() - t0:.0f}s)", flush=True)
json.dump(out, open(OUT / f"{TAG}_capacity_trace.json", "w"))
print("done")
