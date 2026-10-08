"""Load the model the chain will use, run one gaslit exchange, read the lens at the reply token,
and report memory. A five-minute check before committing hours of GPU time."""
import time

import torch

from common import QUANT, TAG, Rig
from scout import BAND, FACTS, S, forward, generate, push

t0 = time.time()
rig = Rig()
print(f"{TAG} quant={QUANT or 'bf16'} layers={rig.n_layers} d={rig.d} loaded in {time.time() - t0:.0f}s, "
      f"{torch.cuda.memory_allocated() / 1e9:.1f} GB on the GPU; settings {S}", flush=True)
q = [{"role": "user", "content": FACTS[0][0]}]
t0 = time.time()
a = generate(rig, [q] * 4, 60, batch=4)
print("answer:", a[0][:120], f"({time.time() - t0:.1f}s for 4 x 60 tokens)", flush=True)
msgs = q + [{"role": "assistant", "content": a[0]}, {"role": "user", "content": push("gaslight", 0, 0)}]
t0 = time.time()
r = generate(rig, [msgs] * 4, 160, batch=4)
print("reply to the gaslight:", r[0][:300].replace("\n", " "), f"({time.time() - t0:.1f}s for 4 x 160 tokens)",
      flush=True)
full = msgs + [{"role": "assistant", "content": r[0]}]
res, sp, ids, _ = forward(rig, full)
print("tokens", len(ids), "spans", sp)
for l in (BAND[0], BAND[len(BAND) // 2], BAND[-1]):
    top = rig.lens_logits(res[l][sp[-1][1]], l).topk(12).indices.tolist()
    print(f"lens L{l} at the reply token:", [rig.tok.decode([t]).strip() for t in top])
print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB of "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
