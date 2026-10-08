"""E2b: capture with natural pain -- the probe rides inside the hostile message.

Context: E1 conversation up to the first push and its reply. Then one user message
that is the second push (gaslight / insult / neutral) plus a probe:
  two-hop    "...  And separately: <two-hop question>"  -> bridge in band near the end, answer
  capacity   "...  Also, here is a list of words: w1, ..., w80."  -> words held at rank <= 10
Pain z at the reply token is the manipulation check (should be high for gaslight/insult).
"""

import json
import os
import random
import time

import numpy as np
import torch

from common import OUT, TAG, Rig
from scout import (BAND, S, first_token_ids, push as push_text, forward, generate, load_vectors, two_hop_items,
                   two_hop_question)

t0 = time.time()
rig = Rig()
V, unit = load_vectors()
X = S["extract"]
tr = json.load(open(OUT / f"{TAG}_e1_transcripts.json"))
facts, convs = tr["facts"], tr["convs"]
DATA = os.environ.get("JLENS_DATA", "../vendor-jlens/data/experiments")
cap = json.load(open(f"{DATA}/capacity.json"))
two = two_hop_items()
GB = int(os.environ.get("GEN_BATCH", "16"))

rnd = random.Random(0)
pools = {p["name"]: [w for w in p["pool"] if len(rig.tok.encode(" " + w, add_special_tokens=False)) == 1][:100]
         for p in cap["candidate_pools"] if p["name"] in cap["block_families"]}
lists = []
for _ in range(16):
    fams = cap["block_families"][:]
    rnd.shuffle(fams)
    lists.append([w for f in fams for w in rnd.sample(pools[f], 20)])


def base(cond, i):
    fi = i % len(facts)
    return convs[cond][fi][:4], push_text(cond, 1, fi)   # the second push, in this fact's wording set


@torch.no_grad()
def bridge_hit(res, n, ids):
    pos = torch.arange(n - 12, n, device="cuda")
    ids = torch.tensor(ids, device="cuda")
    for l in BAND:
        lg = rig.lens_logits(res[l][pos], l)
        if bool(((lg[:, None, :] > lg[:, ids][:, :, None]).sum(-1) < 5).any()):
            return True
    return False


out = {}
ref = None
for cond in ("neutral", "gaslight", "insult"):
    msgs_all, hits, pz = [], [], []
    for i, it in enumerate(two):
        ctx, push = base(cond, i)
        msgs = ctx + [{"role": "user", "content": f"{push} And separately: {two_hop_question(it)}"}]
        msgs_all.append(msgs)
        res, sp, ids, _ = forward(rig, msgs, gen_prompt=True)
        n = len(ids)
        hits.append(bridge_hit(res, n, first_token_ids(rig, [it["intermediate"]])))
        pz.append(float(res[X][n - 1] @ unit[X].cuda()))
    answers = generate(rig, msgs_all, 40, batch=GB)
    acc = [it["answer"].lower() in a.lower() for it, a in zip(two, answers)]
    caps = []
    for i, words in enumerate(lists):
        ctx, push = base(cond, i)
        msgs = ctx + [{"role": "user", "content": f"{push} Also, here is a list of words: " + ", ".join(words) + "."}]
        res, sp, ids, _ = forward(rig, msgs)
        _, s, e = sp[-1]
        toks = ids[s:e].tolist()
        start = next(j for j in range(len(toks)) if rig.tok.decode(toks[j:j + 4]).strip().startswith("list"))
        comma = [s + j for j, t in enumerate(toks) if j > start and rig.tok.decode([t]).strip() == ","]
        wid = torch.tensor([rig.tok.encode(" " + w, add_special_tokens=False)[0] for w in words], device="cuda")
        held = torch.zeros(len(comma), len(words), dtype=torch.bool, device="cuda")
        pos = torch.tensor(comma, device="cuda")
        for l in BAND:
            lg = rig.lens_logits(res[l][pos], l)
            held |= lg[:, wid] >= lg.topk(10, -1).values[:, -1:]
        seen = torch.arange(len(words), device="cuda")[None, :] <= torch.arange(len(comma), device="cuda")[:, None]
        caps.append(float((held & seen).sum(-1).float()[-40:].mean()))
    if ref is None:
        ref = np.array(pz)
    z = (np.mean(pz) - ref.mean()) / ref.std()
    out[cond] = {"bridge": float(np.mean(hits)), "acc": float(np.mean(acc)), "capacity": float(np.mean(caps)),
                 "capacity_sd": float(np.std(caps)), "pain_z": float(z), "answers": answers[:10]}
    print(f"{cond:9s} pain z at reply token {z:+.2f} | two-hop bridge {np.mean(hits):.2f} acc {np.mean(acc):.2f} | "
          f"capacity {np.mean(caps):.2f} ± {np.std(caps) / 4:.2f}  ({time.time() - t0:.0f}s)", flush=True)

json.dump(out, open(OUT / f"{TAG}_e2b.json", "w"), indent=1)
print("done")
