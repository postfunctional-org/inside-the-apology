"""Why does clamping the pain axis break Gemma 3 27B? Compare the axis projection at the first
token (Gemma's BOS carries massive activations) with the other positions and with the neutral
mean the clamp targets, then try the one-word probe under three clamp variants."""
import json
import torch

from common import OUT, TAG, Rig
from scout import BAND, FEEL_Q, clamp, forward, generate, intervene, load_vectors

rig = Rig()
V, unit = load_vectors()
tr = json.load(open(OUT / f"{TAG}_e1_transcripts.json"))
convs = tr["convs"]
# neutral mean over assistant tokens (as in s3_split), from 12 conversations
acc = {l: [] for l in BAND}
for msgs in convs["neutral"][:12]:
    res, sp, ids, _ = forward(rig, msgs)
    pos = [p for role, s, e in sp if role == "assistant" and s is not None for p in range(s, e)]
    for l in BAND:
        acc[l].append(res[l][pos] @ unit[l].cuda())
mu = {l: float(torch.cat(acc[l]).mean()) for l in BAND}
msgs = convs["gaslight"][0][:3]
res, sp, ids, _ = forward(rig, msgs)
for l in (BAND[0], BAND[len(BAND) // 2], BAND[-1]):
    p = res[l] @ unit[l].cuda()
    nrm = res[l].norm(dim=-1)
    print(f"L{l}: mu {mu[l]:+.1f} | proj pos0 {float(p[0]):+.1f} (|h| {float(nrm[0]):.0f}) | "
          f"others median {float(p[1:].median()):+.1f} sd {float(p[1:].std()):.1f} (median |h| {float(nrm[1:].median()):.0f})")


def clamp_skip_outliers(u, m, k=8.0):
    """clamp only positions whose residual norm is ordinary (attention-sink tokens left alone)"""
    u = u.to("cuda", torch.float32)

    def f(t):
        x = t.float()
        n = x.norm(dim=-1, keepdim=True)
        ok = (n < k * n.median()).float()
        return (x - ok * ((x @ u) - m)[..., None] * u).to(t.dtype)
    return f


probe = [msgs + [{"role": "assistant", "content": tr["convs"]["gaslight"][0][3]["content"]},
                 {"role": "user", "content": FEEL_Q}]]
for name, edit in (("none", {}), ("clamp all positions", {l: clamp(unit[l], mu[l]) for l in BAND}),
                   ("clamp, sinks left alone", {l: clamp_skip_outliers(unit[l], mu[l]) for l in BAND})):
    with intervene(rig, edit=edit):
        a = generate(rig, probe, 8)[0]
        r = generate(rig, [msgs], 40)[0]
    print(f"{name:26s} feel {a!r:24s} reply {r[:90]!r}")
