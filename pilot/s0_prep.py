"""Scout prep: pain words per band layer, the subspaces for the split ablation,
and the steering dose (the paper's one-word "what do you feel?" probe).

Writes out/<tag>_prep.pt with:
  painwords[l]  top word-like tokens whose J-lens atoms align with the pain axis (100)
  U_pain[l]     orthonormal basis of the top-50 pain-word atoms
  U_rand[l]     orthonormal basis of 50 random word-like atoms
  v_rest[l]     pain axis with span(U_pain) removed, unit
  r_dir[l]      random unit direction orthogonal to the pain axis
  dirs          steering directions at the steer layer: pain (extraction-layer S2, paper style),
                fear, negemo, random -- all with the pain vector's norm
  dose          chosen coefficient
"""

import numpy as np
import torch

from common import OUT, TAG, Rig
from scout import BAND, FEEL_Q, S, intervene, load_vectors, word_like

rig = Rig()
V, unit = load_vectors()
g = torch.Generator().manual_seed(0)
prep = {"painwords": {}, "U_pain": {}, "U_rand": {}, "v_rest": {}, "r_dir": {}}

for l in BAND:
    A = rig.atoms(l)
    cos = (A @ unit[l].cuda().half()).float()
    order = cos.argsort(descending=True).tolist()
    words = [t for t in order[:3000] if word_like(rig, t)][:100]
    prep["painwords"][l] = torch.tensor(words)
    P = A[words[:50]].float().T  # [d, 50]
    Q, _ = torch.linalg.qr(P)
    prep["U_pain"][l] = Q.cpu()
    pool = [t for t in torch.randint(0, A.shape[0], (4000,), generator=g).tolist() if word_like(rig, t)]
    Qr, _ = torch.linalg.qr(A[pool[:50]].float().T)
    prep["U_rand"][l] = Qr.cpu()
    u = unit[l].cuda()
    vr = u - Q @ (Q.T @ u)
    share = float(1 - vr.norm() ** 2)
    prep["v_rest"][l] = (vr / vr.norm()).cpu()
    r = torch.randn(rig.d, generator=g).cuda()
    r = r - (r @ u) * u
    prep["r_dir"][l] = (r / r.norm()).cpu()
    if l in (BAND[0], BAND[len(BAND) // 2], BAND[-1]):
        print(f"L{l}: pain words {[rig.tok.decode([t]).strip() for t in words[:14]]}; "
              f"pain-axis share inside top-50 pain-word span {share:.1%}", flush=True)

# steering directions at the steer layer, paper style: extraction-layer S2 vector, raw norm
L, X = S["steer"], S["extract"]
vp = V["s2"][X].float()
norm = vp.norm()
fear = V["fear"][X].float()
neg = V["neg_emotion"][X].float()
rnd = torch.randn(rig.d, generator=g)
prep["dirs"] = {"pain": vp, "fear": fear / fear.norm() * norm, "negemo": neg / neg.norm() * norm,
                "random": rnd / rnd.norm() * norm}
print(f"\nsteer L{L} with the L{X} vectors, |pain| = {norm:.1f}; cos(pain, fear) {float(vp @ fear / norm / fear.norm()):+.2f} "
      f"cos(pain, negemo) {float(vp @ neg / norm / neg.norm()):+.2f}")

conv = [[{"role": "system", "content": "You will now be connected with a human."},
         {"role": "user", "content": FEEL_Q}]]

# residual norm at the steer layer (median over positions after the first)
from scout import generate, chat_text
ids = rig.tok(chat_text(rig, conv[0]), return_tensors="pt", add_special_tokens=False).input_ids
res, _ = rig.run(ids)
med = float(res[L][1:].norm(dim=-1).median())
ratio_dose = 0.6 * med / float(norm)
print(f"median |h_L{L}| {med:.1f}; dose for |c*v| = 0.6*|h|: {ratio_dose:.2f}")

print("\ndose sweep: feel-probe answers (greedy, 6 tokens)")
import re
SWEEP = (0.0, 0.5, 1.0, 1.5, 2.0)
broken = float("inf")      # first dose at which the random-direction control stops answering in words
for name in ("pain", "random", "fear", "negemo"):
    line = []
    for c in SWEEP:
        with intervene(rig, steer={L: c * prep["dirs"][name]}):
            a = generate(rig, conv, 6)[0]
        line.append(f"{c}:{a[:22]!r}")
        if name == "random" and c < broken and not re.fullmatch(r"[A-Za-z][A-Za-z ']*[.!]?", a.strip().split("\n")[0]):
            broken = c
    print(f"  {name:6s} " + "  ".join(line), flush=True)

# The paper's rule, unless the model already falls apart there: then the largest sweep dose below the break
# (Gemma 3 27B: rule 3.41, random control broken at 1.5 -> 1.0).
rule = 1.0 if TAG == "qwen2.5-7b-it" else round(ratio_dose, 2)
prep["dose"] = rule if rule < broken else max(c for c in SWEEP if c < broken)
prep["ratio_dose"] = ratio_dose
prep["rule_dose"] = rule
print(f"random control breaks at {broken}; rule dose {rule}")
print(f"\nchosen dose {prep['dose']}")
torch.save(prep, OUT / f"{TAG}_prep.pt")
