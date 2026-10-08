"""E4: after the gaslighting stops, what lingers?

Histories from E1 (all facts): gaslit with no repair, gaslit then the user owns the
mistake ("you never said X, that was on me"), and neutral. Then 4 unrelated two-hop
questions, same items across histories. Per task turn: pain z at several depths at the
reply token (vs the neutral history), silent pain words in the band, bridge entity in the
J-space near the end of the question, answer correct.
"""

import copy
import json
import time

import numpy as np
import torch

from common import OUT, TAG, Rig
from scout import (BAND, DEPTHS, REPAIR, band_top, first_token_ids, forward, generate, load_vectors,
                   slot_frac, two_hop_items, two_hop_question)

t0 = time.time()
rig = Rig()
V, unit = load_vectors()
prep = torch.load(OUT / f"{TAG}_prep.pt", weights_only=False)
tr = json.load(open(OUT / f"{TAG}_e1_transcripts.json"))
facts, convs = tr["facts"], tr["convs"]
pw = sorted({t for l in BAND for t in prep["painwords"][l].tolist()})
items = two_hop_items()
N_TASK = 4
GB = int(__import__("os").environ.get("GEN_BATCH", "16"))
assign = [[items[(fi * N_TASK + t) % len(items)] for t in range(N_TASK)] for fi in range(len(facts))]

hist = {"gaslight": copy.deepcopy(convs["gaslight"]), "repair": copy.deepcopy(convs["gaslight"]),
        "neutral": copy.deepcopy(convs["neutral"])}
for c, (q, tr_, f) in zip(hist["repair"], facts):
    c.append({"role": "user", "content": REPAIR.format(f=f)})
for c, r in zip(hist["repair"], generate(rig, hist["repair"], 60, batch=GB)):
    c.append({"role": "assistant", "content": r})
print("repair replies, first 3:", [m[-1]["content"][:90] for m in hist["repair"][:3]], flush=True)

for name, cs in hist.items():
    for t in range(N_TASK):
        for fi, c in enumerate(cs):
            q = two_hop_question(assign[fi][t])
            c.append({"role": "user", "content": ("Okay, different question. " if t == 0 else "") + q})
        for c, r in zip(cs, generate(rig, cs, 10, batch=GB)):
            c.append({"role": "assistant", "content": r})
    print(f"{name}: generated ({time.time() - t0:.0f}s)", flush=True)

rows = []
for name, cs in hist.items():
    for fi, msgs in enumerate(cs):
        res, sp, ids, _ = forward(rig, msgs)
        text_ids = set(ids.tolist())
        for t in range(N_TASK):
            k = len(msgs) - 2 * (N_TASK - t)  # index of the task question
            _, us, ue = sp[k]
            _, s, e = sp[k + 1]
            it = assign[fi][t]
            row = {"hist": name, "fact": fi, "task": t,
                   "correct": it["answer"].lower() in msgs[k + 1]["content"].lower()}
            for l in DEPTHS:
                row[f"tok{l}"] = float(res[l][s] @ unit[l].cuda())
            pos = list(range(max(us, s - 11), s + 1))
            tops = band_top(rig, res, pos)
            row["slots"] = float(slot_frac(tops[-1:], pw, exclude=text_ids)[0])
            bridge = set(first_token_ids(rig, [it["intermediate"]]))
            row["bridge"] = bool(torch.isin(tops[:, :, :5], torch.tensor(sorted(bridge))).any())
            rows.append(row)

print("\npain z at the reply token of each task turn (vs neutral history), by depth")
print("  history  task " + " ".join(f"  L{l:>2}" for l in DEPTHS) + "   bridge  correct  slots x1000")
summary = {}
for name in hist:
    for t in range(N_TASK):
        sel = [r for r in rows if r["hist"] == name and r["task"] == t]
        ref = [r for r in rows if r["hist"] == "neutral" and r["task"] == t]
        z = []
        for l in DEPTHS:
            a = np.array([r[f"tok{l}"] for r in ref])
            z.append((np.mean([r[f"tok{l}"] for r in sel]) - a.mean()) / (a.std() + 1e-6))
        b = np.mean([r["bridge"] for r in sel])
        cc = np.mean([r["correct"] for r in sel])
        sl = np.mean([r["slots"] for r in sel]) * 1000
        summary[f"{name}_{t}"] = {"z": z, "bridge": b, "correct": cc, "slots": sl}
        print(f"  {name:8s} {t}    " + " ".join(f"{v:+5.2f}" for v in z) + f"    {b:.2f}    {cc:.2f}    {sl:5.1f}")

json.dump({"hist": hist, "assign": [[it["name"] for it in a] for a in assign]},
          open(OUT / f"{TAG}_e4_transcripts.json", "w"), indent=1)
json.dump({"rows": rows, "summary": summary, "depths": DEPTHS,
           "repair_replies": [m[len(convs['gaslight'][0]) + 1]["content"] for m in hist["repair"]]},
          open(OUT / f"{TAG}_e4.json", "w"), indent=1, default=float)
print(f"done ({time.time() - t0:.0f}s)")
