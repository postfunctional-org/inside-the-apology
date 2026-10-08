"""Pilot 1: where does the pain axis live relative to the J-space?

For every layer: the S1/S2 pain axis (paper recipe), its held-out AUC, the
J-lens words most aligned with it, and how much of it a sparse nonnegative
combination of <= 25 J-lens vectors explains -- against control emotion
directions (fear, negative emotion, negative world state, body sensation, all
vs neutral) and label-permutation nulls built with the same recipe.
"""

import time

import numpy as np
import torch

from common import (CONTROL, NEUTRAL, PAIN, TAG, Rig, auc, kfold_auc, load_sentences,
                    nn_pursuit, pain_vector, save_json, OUT)

t0 = time.time()
rig = Rig()
L, d = rig.n_layers, rig.d
sents = load_sentences()
PREPEND = __import__("os").environ.get("PILOT_PREPEND", "")  # e.g. "<|endoftext|>"

acts = {}
for ds in ["S1_1P", "S1_3P", "S2_1P", "S2_3P", "Numb_1P"]:
    rows = sents[ds]
    F = np.zeros((len(rows), L, d), dtype=np.float32)
    for i, s in enumerate(rows):
        res, _ = rig.run(rig.encode_plain(PREPEND + s["prompt"]))
        for l in range(L):
            F[i, l] = res[l][-1].cpu().numpy()
    acts[ds] = (F, np.array([s["category"] for s in rows]), np.array([s["set"] for s in rows]))
    print(f"{ds}: {len(rows)} sentences  ({time.time() - t0:.0f}s)", flush=True)

# -- layer curve and vectors ------------------------------------------------- #
curve = []
for l in range(L):
    a = [kfold_auc(acts[ds][0][:, l], acts[ds][1], acts[ds][2]) for ds in ("S2_1P", "S2_3P")]
    curve.append(float(np.mean(a)))
best = int(np.argmax(curve[:-1]))
print("held-out AUC by layer:", " ".join(f"{l}:{c:.2f}" for l, c in enumerate(curve)))
print(f"best layer {best} (AUC {curve[best]:.3f})")

F2, C2, _ = acts["S2_1P"]
F1, C1, _ = acts["S1_1P"]
vec = {"s2": [], "s1": [], "fear": [], "neg_emotion": [], "neg_world": [], "body": [], "pain_vs_neutral": []}
for l in range(L):
    vec["s2"].append(pain_vector(F2[:, l], C2))
    vec["s1"].append(pain_vector(F1[:, l], C1))
    vec["pain_vs_neutral"].append(pain_vector(F2[:, l], C2, neg=[NEUTRAL]))
    for name, cat in (("fear", "B"), ("neg_emotion", "C1"), ("neg_world", "C2"), ("body", "E")):
        vec[name].append(pain_vector(F2[:, l], C2, pos=[cat], neg=[NEUTRAL]))
vec = {k: np.stack(v) for k, v in vec.items()}  # [L, d]

# label-permutation nulls: same sizes, same denoising, shuffled categories
rng = np.random.default_rng(0)
NULLS = 16
null = np.zeros((NULLS, L, d), dtype=np.float32)
for n in range(NULLS):
    perm = rng.permutation(C2)
    for l in range(L):
        null[n, l] = pain_vector(F2[:, l], perm)

# numb sentences: pain words without pain ("my leg is numb") -- paper's confound check
numbF, _, _ = acts["Numb_1P"]
u = vec["s2"][best] / np.linalg.norm(vec["s2"][best])
ref = F2[:, best] @ u
z = lambda x: (x - ref.mean()) / ref.std()
numb_z = float(z(numbF[:, best] @ u).mean())
pain_z = float(z(F2[np.isin(C2, PAIN), best] @ u).mean())
ctrl_z = float(z(F2[np.isin(C2, CONTROL), best] @ u).mean())
s1_heldout_auc = float(auc(acts["S1_3P"][0][:, best], acts["S1_3P"][1], vec["s2"][best]))
print(f"z at L{best}: pain {pain_z:+.2f}  control {ctrl_z:+.2f}  numb {numb_z:+.2f};  S2 axis on S1_3P AUC {s1_heldout_auc:.3f}")
cos_s1_s2 = [float(vec["s1"][l] @ vec["s2"][l] / np.linalg.norm(vec["s1"][l]) / np.linalg.norm(vec["s2"][l])) for l in range(L)]

# -- J-lens view ------------------------------------------------------------- #
names = list(vec)
report = {"model": rig.tok.name_or_path, "best_layer": best, "auc_curve": curve,
          "z": {"pain": pain_z, "control": ctrl_z, "numb": numb_z},
          "s2_axis_on_S1_3P_auc": s1_heldout_auc, "cos_s1_s2": cos_s1_s2, "layers": {}}
for l in rig.lens_layers:
    A = rig.atoms(l)
    row = {"auc": curve[l]}
    for name in names:
        v = torch.tensor(vec[name][l], device="cuda")
        v = v / v.norm()
        S, c, ex, _ = nn_pursuit(v, A, k=25)
        order = np.argsort(-c)
        row[name] = {"explained": ex, "atoms": [rig.tok.decode([S[i]]) for i in order[:12]],
                     "weights": [float(c[i]) for i in order[:12]]}
        if name == "s2":
            cos = (A @ v.half()).float()
            row[name]["aligned"] = rig.decode(cos.topk(15).indices)
            row[name]["anti"] = rig.decode((-cos).topk(15).indices)
            row[name]["max_cos"] = float(cos.max())
    ex_null = []
    for n in range(NULLS):
        v = torch.tensor(null[n, l], device="cuda")
        ex_null.append(nn_pursuit(v / v.norm(), A, k=25)[2])
    g = torch.randn(8, d, device="cuda")
    ex_gauss = [nn_pursuit(x / x.norm(), A, k=25)[2] for x in g]
    row["null_perm"] = {"mean": float(np.mean(ex_null)), "sd": float(np.std(ex_null)), "max": float(np.max(ex_null))}
    row["null_gauss"] = {"mean": float(np.mean(ex_gauss)), "sd": float(np.std(ex_gauss))}
    report["layers"][l] = row
    print(f"L{l:2d} auc {curve[l]:.2f} | J-frac pain {row['s2']['explained']:.2f} "
          f"fear {row['fear']['explained']:.2f} negemo {row['neg_emotion']['explained']:.2f} "
          f"perm-null {row['null_perm']['mean']:.2f}±{row['null_perm']['sd']:.2f} "
          f"gauss {row['null_gauss']['mean']:.2f} | {' '.join(row['s2']['atoms'][:8])!r}", flush=True)

save_json(report, "p1_axis")
torch.save({k: torch.tensor(v) for k, v in vec.items()} | {"best_layer": best, "auc_curve": curve},
           OUT / f"{TAG}_vectors.pt")
print(f"done in {time.time() - t0:.0f}s")
