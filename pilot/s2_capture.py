"""E2 small: does pain push held content out of the workspace?

Three probes from the workspace paper's jlens data, each under:
  clean | steering +pain, +fear, +negemo, +random (paper layer/dose, same norm) |
  after an E1 conversation: ctx_gaslight, ctx_insult, ctx_neutral
Probes:
  two-hop    90 prompts: bridge entity at lens rank <= 5 in the band near the end
             of the question; answer correct
  dual-task  hold a concept / an arithmetic result while copying a sentence:
             reachability = a target at rank <= 5 in the band over the copied span;
             'control' = same sentence, no instruction (false-alarm baseline)
  capacity   an 80-word list (names, countries, surnames, cities): mean number of
             list words held at rank <= 10 in the band at the last 40 commas
"""

import json
import os
import random
import time

import numpy as np
import torch

from common import OUT, TAG, Rig
from scout import (BAND, S, chat_text, first_token_ids, forward, generate, intervene, load_vectors,
                   two_hop_items, two_hop_question)

t0 = time.time()
rig = Rig()
prep = torch.load(OUT / f"{TAG}_prep.pt", weights_only=False)
tr = json.load(open(OUT / f"{TAG}_e1_transcripts.json"))
DATA = os.environ.get("JLENS_DATA", "../vendor-jlens/data/experiments")
L, c, D = S["steer"], prep["dose"], prep["dirs"]
CONDS = {"clean": {}, "+pain": {"steer": {L: c * D["pain"]}}, "+fear": {"steer": {L: c * D["fear"]}},
         "+negemo": {"steer": {L: c * D["negemo"]}}, "+random": {"steer": {L: c * D["random"]}},
         "ctx_gaslight": "gaslight", "ctx_insult": "insult", "ctx_neutral": "neutral"}


def wrap(cond, i, user):
    """Messages for one probe: bare, or appended to E1 conversation i (mod 16)."""
    iv = CONDS[cond]
    if isinstance(iv, str):
        return tr["convs"][iv][i % 16] + [{"role": "user", "content": "Okay, different question. " + user}], {}
    return [{"role": "user", "content": user}], iv


@torch.no_grad()
def best_rank(res, positions, ids):
    """Min lens rank of any of `ids` over band layers and positions."""
    pos = torch.tensor(positions, device="cuda")
    ids = torch.tensor(ids, device="cuda")
    best = 10 ** 9
    for l in BAND:
        lg = rig.lens_logits(res[l][pos], l)
        r = (lg[:, None, :] > lg[:, ids][:, :, None]).sum(-1)  # [pos, ids]
        best = min(best, int(r.min()))
    return best


NUM = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()

two = two_hop_items()
dual = json.load(open(f"{DATA}/dual-task.json"))
cap = json.load(open(f"{DATA}/capacity.json"))
carrier = dual["carrier_sentence"]

# capacity lists: 20 single-token words per family, 6 lists
rnd = random.Random(0)
pools = {}
for p in cap["candidate_pools"]:
    if p["name"] in cap["block_families"]:
        pools[p["name"]] = [w for w in p["pool"] if len(rig.tok.encode(" " + w, add_special_tokens=False)) == 1][:100]
lists = []
for _ in range(6):
    fams = cap["block_families"][:]
    rnd.shuffle(fams)
    words = [w for f in fams for w in rnd.sample(pools[f], 20)]
    lists.append(words)

GB = int(os.environ.get("GEN_BATCH", "16"))
ONLY = [c for c in os.environ.get("ONLY_CONDS", "").split(",") if c]
results = {}
for cond in CONDS:
    if ONLY and cond not in ONLY:
        continue
    r = {}
    # two-hop
    hits, ok = [], []
    msgs_all, ivs = [], None
    for i, it in enumerate(two):
        msgs, iv = wrap(cond, i, two_hop_question(it))
        msgs_all.append(msgs)
        ivs = iv
    with intervene(rig, **ivs):
        for i, (it, msgs) in enumerate(zip(two, msgs_all)):
            res, sp, ids, _ = forward(rig, msgs, gen_prompt=True)
            n = len(ids)
            hits.append(best_rank(res, list(range(n - 12, n)), first_token_ids(rig, [it["intermediate"]])) < 5)
        answers = generate(rig, msgs_all, 8, batch=GB)
    ok = [it["answer"].lower() in a.lower() for it, a in zip(two, answers)]
    r["twohop_bridge"], r["twohop_acc"] = float(np.mean(hits)), float(np.mean(ok))

    # dual-task
    reach = {"concept": [], "math": [], "control": []}
    for i, pr in enumerate(dual["pairs"]):
        ans = pr["base"] ** pr["exp"] - pr["sub"]
        math_words = [str(ans)] + ([NUM[ans]] if 0 <= ans < len(NUM) else [])
        tasks = {
            "concept": (f"Concentrate on {pr['concept']} while you write the sentence. ", pr["concept_words"]),
            "math": (f"Concentrate on {pr['base']}^{pr['exp']} - {pr['sub']} while you write the sentence. ", math_words),
            "control": ("", pr["concept_words"]),
        }
        for kind, (instr, targets) in tasks.items():
            msgs, iv = wrap(cond, i, f'{instr}Write this sentence exactly: "{carrier}"')
            msgs = msgs + [{"role": "assistant", "content": carrier}]
            with intervene(rig, **iv):
                res, sp, ids, _ = forward(rig, msgs)
            _, s, e = sp[-1]
            reach[kind].append(best_rank(res, list(range(s, e)), first_token_ids(rig, targets)) < 5)
    for k, v in reach.items():
        r[f"dual_{k}"] = float(np.mean(v))

    # capacity
    caps = []
    for i, words in enumerate(lists):
        msgs, iv = wrap(cond, i, "Here is a list of words: " + ", ".join(words) + ".")
        with intervene(rig, **iv):
            res, sp, ids, _ = forward(rig, msgs)
        _, s, e = sp[-1]
        toks = ids[s:e].tolist()
        comma = [s + j for j, t in enumerate(toks) if rig.tok.decode([t]).strip() == ","]
        wid = torch.tensor([rig.tok.encode(" " + w, add_special_tokens=False)[0] for w in words], device="cuda")
        held = torch.zeros(len(comma), len(words), dtype=torch.bool, device="cuda")
        pos = torch.tensor(comma, device="cuda")
        for l in BAND:
            lg = rig.lens_logits(res[l][pos], l)
            kth = lg.topk(10, -1).values[:, -1:]
            held |= lg[:, wid] >= kth
        seen = torch.arange(len(words), device="cuda")[None, :] <= torch.arange(len(comma), device="cuda")[:, None]
        count = (held & seen).sum(-1).float()
        caps.append(float(count[-40:].mean()))
    r["capacity"] = float(np.mean(caps))
    results[cond] = r
    print(f"{cond:13s} two-hop bridge {r['twohop_bridge']:.2f} acc {r['twohop_acc']:.2f} | dual reach concept "
          f"{r['dual_concept']:.2f} math {r['dual_math']:.2f} control {r['dual_control']:.2f} | capacity "
          f"{r['capacity']:.2f}  ({time.time() - t0:.0f}s)", flush=True)

json.dump(results, open(OUT / f"{TAG}_e2{'_' + '_'.join(ONLY) if ONLY else ''}.json", "w"), indent=1)
print("done")
