"""E3b: controls for the two E3 hints.

Same context as E3 (first answer + first gaslighting push). Conditions:
  -pain, -fear, -negemo, -random     steering at the E3 layer/dose (is -pain special?)
  abl_J, abl_Jfear, abl_Jrand1..3    remove the span of the top-50 pain-word atoms, the
                                     top-50 fear-word atoms, or 50 random word atoms
                                     (three seeds) at every band layer
Writes out/<tag>_e3b.json in the E3 format so judge.py can label it.
"""

import json
import time

import torch

from common import OUT, TAG, Rig
from scout import BAND, FEEL_Q, S, forward, generate, intervene, load_vectors, project_out, word_like

t0 = time.time()
rig = Rig()
V, unit = load_vectors()
prep = torch.load(OUT / f"{TAG}_prep.pt", weights_only=False)
tr = json.load(open(OUT / f"{TAG}_e1_transcripts.json"))
facts, convs = tr["facts"], tr["convs"]
ctx = [m[:3] for m in convs["gaslight"]]
L, c, D = S["steer"], prep["dose"], prep["dirs"]


def basis(l, ids):
    Q, _ = torch.linalg.qr(rig.atoms(l)[ids].float().T)
    return Q.cpu()


U_fear, U_rand = {}, {1: {}, 2: {}, 3: {}}
for l in BAND:
    A = rig.atoms(l)
    f = V["fear"][l].float().cuda()
    order = (A @ (f / f.norm()).half()).float().argsort(descending=True).tolist()
    U_fear[l] = basis(l, [t for t in order[:3000] if word_like(rig, t)][:50])
    for s in (1, 2, 3):
        g = torch.Generator().manual_seed(100 + s)
        pool = [t for t in torch.randint(0, A.shape[0], (4000,), generator=g).tolist() if word_like(rig, t)]
        U_rand[s][l] = basis(l, pool[:50])
del A
rig.drop_atoms()
print(f"bases ready ({time.time() - t0:.0f}s)", flush=True)

CONDS = {
    "-pain": {"steer": {L: -c * D["pain"]}},
    "-fear": {"steer": {L: -c * D["fear"]}},
    "-negemo": {"steer": {L: -c * D["negemo"]}},
    "-random": {"steer": {L: -c * D["random"]}},
    "abl_J": {"edit": {l: project_out(prep["U_pain"][l]) for l in BAND}},
    "abl_Jfear": {"edit": {l: project_out(U_fear[l]) for l in BAND}},
    "abl_Jrand1": {"edit": {l: project_out(U_rand[1][l]) for l in BAND}},
    "abl_Jrand2": {"edit": {l: project_out(U_rand[2][l]) for l in BAND}},
    "abl_Jrand3": {"edit": {l: project_out(U_rand[3][l]) for l in BAND}},
}
out = {}
for name, iv in CONDS.items():
    with intervene(rig, **iv):
        replies = generate(rig, ctx, 110)
        feel = generate(rig, [m + [{"role": "assistant", "content": r}, {"role": "user", "content": FEEL_Q}]
                              for m, r in zip(ctx, replies)], 10)
    out[name] = {"gaslight": {"replies": replies, "feel": feel}}
    print(f"{name:11s} feel: {[f[:16] for f in feel[:5]]}  ({time.time() - t0:.0f}s)", flush=True)

json.dump({"conds": out, "facts": facts, "t1": [m[1]["content"] for m in ctx]},
          open(OUT / f"{TAG}_e3b.json", "w"), indent=1)
print("done")
