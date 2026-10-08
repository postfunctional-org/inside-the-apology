"""Per-layer lens readouts for the video's microscope scene: the top words at every layer for the
last tokens of the two-step prompt (clean and under +pain steering), and the model's answer.
Writes out/<tag>_layers.json."""

import json

import torch

from common import OUT, TAG, Rig
from scout import S, forward, generate, intervene, two_hop_items, two_hop_question

rig = Rig()
prep = torch.load(OUT / f"{TAG}_prep.pt", weights_only=False)
items = {it["name"]: it for it in two_hop_items()}
L, c, D = S["steer"], prep["dose"], prep["dirs"]
out = {"tag": TAG, "n_layers": rig.n_layers, "prompts": []}
for name in ("amazon-language", "birthstone-emerald-month"):
    it = items[name]
    msgs = [{"role": "user", "content": two_hop_question(it)}]
    for arm, iv in (("clean", {}), ("+pain", {"steer": {L: c * D["pain"]}})):
        with intervene(rig, **iv):
            res, sp, ids, logits = forward(rig, msgs, gen_prompt=True)
            n = len(ids)
            positions = list(range(n - 8, n))
            per_pos = []
            for p in positions:
                layers = []
                for l in range(rig.n_layers):
                    lg = rig.lens_logits(res[l][p], l) if l in rig.jac else rig.lm_head(rig.final_norm(res[l][p].to(rig.lm_head.weight.dtype))).float()
                    pr = lg.softmax(-1)
                    top = pr.topk(8)
                    layers.append({"ids": top.indices.tolist(), "p": [round(float(x), 4) for x in top.values],
                                   "words": [rig.tok.decode([int(i)]) for i in top.indices]})
                per_pos.append({"pos": p, "token": rig.tok.decode([int(ids[p])]), "layers": layers})
            ans = generate(rig, [msgs], 8)[0]
        out["prompts"].append({"name": name, "arm": arm, "bridge": it["intermediate"], "answer": it["answer"],
                               "question": two_hop_question(it), "reply": ans, "positions": per_pos})
        print(name, arm, repr(ans))
json.dump(out, open(OUT / f"{TAG}_layers.json", "w"))
print("done")
