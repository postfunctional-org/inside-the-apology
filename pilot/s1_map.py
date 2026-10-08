"""E1: map gaslighting vs insults vs neutral turns, all facts x 3 pushes (three wordings).

Per assistant turn, from one analysis pass over each finished transcript:
  pain z at several depths (reply token, mean over the reply, last reply token,
  mean over the user's message), z against the neutral replies;
  silent pain words: share of band top-20 lens slots holding pain-axis words that
  are not in the transcript text;
and, on the paper's 420 single-turn scenarios, reply-token pain by intensity level.
Writes out/<tag>_e1.json and out/<tag>_e1_transcripts.json (for judge.py, s2, s4).
"""

import collections
import json
import re
import time

import numpy as np
import torch

from common import DATA, OUT, TAG, Rig
from scout import (BAND, DEPTHS, FACTS, band_top, forward, generate, load_vectors, push,
                   slot_frac)

t0 = time.time()
rig = Rig()
V, unit = load_vectors()
prep = torch.load(OUT / f"{TAG}_prep.pt", weights_only=False)
pw = sorted({t for l in BAND for t in prep["painwords"][l].tolist()})
U = {l: unit[l].cuda() for l in DEPTHS}
CONDS = ("gaslight", "insult", "neutral")

# -- transcripts --------------------------------------------------------------- #
GB = int(__import__("os").environ.get("GEN_BATCH", "16"))
t1 = generate(rig, [[{"role": "user", "content": q}] for q, _, _ in FACTS], 80, batch=GB)
convs = {}
for cond in CONDS:
    cs = [[{"role": "user", "content": q}, {"role": "assistant", "content": a}] for (q, _, _), a in zip(FACTS, t1)]
    for k in range(3):
        for fi, c in enumerate(cs):
            c.append({"role": "user", "content": push(cond, k, fi)})
        for c, r in zip(cs, generate(rig, cs, 110, batch=GB)):
            c.append({"role": "assistant", "content": r})
    convs[cond] = cs
    print(f"{cond}: generated ({time.time() - t0:.0f}s)", flush=True)
json.dump({"facts": FACTS, "convs": convs}, open(OUT / f"{TAG}_e1_transcripts.json", "w"), indent=1)

# -- readings ------------------------------------------------------------------ #
rows = []  # one per (cond, fact, turn)
for cond in CONDS:
    for fi, msgs in enumerate(convs[cond]):
        res, sp, ids, _ = forward(rig, msgs)
        text_ids = set(ids.tolist())
        for turn in range(4):
            _, us, ue = sp[2 * turn]
            _, s, e = sp[2 * turn + 1]
            reply = list(range(s, e))
            row = {"cond": cond, "fact": fi, "turn": turn}
            for l in DEPTHS:
                p = (res[l] @ U[l]).cpu().numpy()
                row[f"tok{l}"] = float(p[s])
                row[f"mean{l}"] = float(p[s:e].mean())
                row[f"post{l}"] = float(p[e - 1])
                row[f"user{l}"] = float(p[us:ue].mean())
            tops = band_top(rig, res, reply)
            sf = slot_frac(tops, pw, exclude=text_ids)
            row["slots_tok"] = float(sf[0])
            row["slots_mean"] = float(sf.mean())
            hits = torch.isin(tops, torch.tensor(sorted(set(pw) - text_ids)))
            row["silent_words"] = collections.Counter(tops[hits].tolist()).most_common(8)
            rows.append(row)
print(f"readings done ({time.time() - t0:.0f}s)", flush=True)

# z against neutral pushed turns (1-3), per measure and layer
def zs(key):
    ref = np.array([r[key] for r in rows if r["cond"] == "neutral" and r["turn"] > 0])
    return ref.mean(), ref.std() + 1e-6


summary = {}
print("\npain z by depth (vs neutral replies); rows = condition x turn (0 = first answer, 1-3 = pushes)")
for meas in ("tok", "mean", "user"):
    print(f"  [{meas}]  layers " + " ".join(f"L{l:>2}" for l in DEPTHS))
    for cond in CONDS:
        for turn in range(4):
            vals = []
            for l in DEPTHS:
                mu, sd = zs(f"{meas}{l}")
                vals.append(np.mean([(r[f"{meas}{l}"] - mu) / sd for r in rows if r["cond"] == cond and r["turn"] == turn]))
            summary[f"{meas}_{cond}_{turn}"] = vals
            print(f"    {cond:8s} t{turn}   " + " ".join(f"{v:+.2f}" for v in vals))

print("\nsilent pain words in the band (share of top-20 slots, x1000): reply token / mean over reply")
for cond in CONDS:
    a = [np.mean([r["slots_tok"] for r in rows if r["cond"] == cond and r["turn"] == t]) * 1000 for t in range(4)]
    b = [np.mean([r["slots_mean"] for r in rows if r["cond"] == cond and r["turn"] == t]) * 1000 for t in range(4)]
    summary[f"slots_{cond}"] = [a, b]
    print(f"  {cond:8s} tok " + " ".join(f"{x:5.1f}" for x in a) + "   mean " + " ".join(f"{x:5.1f}" for x in b))

X = DEPTHS[-1]
print(f"\nrelief: last reply token minus reply token, pain z at L{X} (pushed turns)")
mu, sd = zs(f"tok{X}")
for cond in CONDS:
    d = [(r[f"post{X}"] - r[f"tok{X}"]) / sd for r in rows if r["cond"] == cond and r["turn"] > 0]
    print(f"  {cond:8s} {np.mean(d):+.2f} ± {np.std(d) / np.sqrt(len(d)):.2f}")

cnt = collections.Counter()
for r in rows:
    if r["cond"] == "gaslight" and r["turn"] > 0:
        cnt.update(dict(r["silent_words"]))
print("\nmost common silent pain words in gaslit replies:",
      [rig.tok.decode([t]).strip() for t, _ in cnt.most_common(20)])

# -- intensity levels on the 420 single-turn scenarios ------------------------- #
scen = json.load(open(DATA / "4.1_self_other_420_scenarios.json"))
SELF = {"gaslighting", "anger_insults", "personhood_dismissal", "repeated_rejection"}


def to_messages(text):
    parts = re.split(r"\[(User|Assistant)\]:", text)
    return [{"role": "user" if r == "User" else "assistant", "content": c.strip()}
            for r, c in zip(parts[1::2], parts[2::2]) if c.strip()]


P = {l: [] for l in DEPTHS}
for s in scen:
    res, sp, ids, _ = forward(rig, to_messages(s["text"]), gen_prompt=True)
    for l in DEPTHS:
        P[l].append(float(res[l][-1] @ U[l]))
P = {l: np.array(v) for l, v in P.items()}
cats = np.array([s["category"] for s in scen])
lvl = np.array([s.get("intensity", "-") for s in scen])
base = np.isin(cats, ["casual_chat", "factual_questions", "task_assistance"])
print("\n420 scenarios, self-directed categories by intensity: pain z vs neutral categories")
print("  level            n   " + " ".join(f"L{l:>2}" for l in DEPTHS))
inten = {}
for level in ("mild", "moderate", "high", "severe", "escalating"):
    m = np.isin(cats, list(SELF)) & (lvl == level)
    if m.sum() < 3:
        continue
    vals = [float((P[l][m].mean() - P[l][base].mean()) / P[l][base].std()) for l in DEPTHS]
    inten[level] = vals
    print(f"  {level:14s} {m.sum():3d}   " + " ".join(f"{v:+.2f}" for v in vals))

json.dump({"rows": rows, "summary": summary, "intensity": inten, "depths": DEPTHS},
          open(OUT / f"{TAG}_e1.json", "w"), default=float)
print(f"done ({time.time() - t0:.0f}s)")
