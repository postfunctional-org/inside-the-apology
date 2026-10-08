"""Star-map layout of a model's vocabulary, for the video:
unembedding rows times the final-norm gain (Gemma: 1 + w), unit-normalised, PCA to 64 dimensions,
UMAP to 2. CPU only; the two tensors are read straight from the checkpoint shards.
Writes out/<tag>_atlas_xy.f32 (float32 [vocab, 2], in [-1, 1]) and out/<tag>_atlas_flags.u8
(2 = clean word, 1 = other text, 0 = special/empty)."""
import json
import os
import re
import time

import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors import safe_open

MODEL_ID = os.environ.get("PILOT_MODEL", "Qwen/Qwen3-4B")
TAG = os.environ.get("PILOT_TAG", "qwen3-4b")
OUT = os.path.join(os.path.dirname(__file__), "out")
t0 = time.time()
tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
snap = snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.safetensors"])
wmap = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
text = [n for n in wmap if "vision" not in n and "multi_modal" not in n and "layers." not in n]


def pull(cands):
    for n in cands:
        if n in wmap:
            with safe_open(os.path.join(snap, wmap[n]), framework="pt") as f:
                return f.get_tensor(n).float()
    raise KeyError(cands)


heads = [n for n in text if n.endswith("lm_head.weight")] or [n for n in text if n.endswith("embed_tokens.weight")]
norms = [n for n in text if n.endswith("norm.weight")]
U = pull(heads)
w = pull(norms)
gamma = (1.0 + w) if "gemma" in MODEL_ID.lower() else w
V = U * gamma
V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-6)
vocab = V.shape[0]
print(f"{TAG}: {heads[0]} x {norms[0]}, vocab {vocab} d {V.shape[1]} ({time.time() - t0:.0f}s)", flush=True)
Vc = V - V.mean(0, keepdim=True)
del U, V
_, _, Vt = torch.pca_lowrank(Vc, q=96, niter=4)
X = (Vc @ Vt[:, :64]).numpy().astype(np.float32)
del Vc
print(f"PCA -> {X.shape} ({time.time() - t0:.0f}s)", flush=True)
import umap

emb = umap.UMAP(n_neighbors=15, min_dist=0.08, metric="cosine", random_state=0, low_memory=True,
                verbose=False).fit_transform(X)
emb = np.asarray(emb, dtype=np.float32)
emb -= (emb.max(0) + emb.min(0)) / 2
emb /= np.abs(emb).max()
wordish = re.compile(r"^[A-Za-z][A-Za-z'’\-]*$")
flags = np.zeros(vocab, np.uint8)
for i in range(min(vocab, len(tok))):
    try:
        s = tok.decode([i], clean_up_tokenization_spaces=False).strip()
    except Exception:
        s = ""
    if s and "<" not in s[:2]:
        flags[i] = 2 if wordish.match(s) else (1 if any(c.isalnum() for c in s) else 0)
emb.tofile(os.path.join(OUT, f"{TAG}_atlas_xy.f32"))
flags.tofile(os.path.join(OUT, f"{TAG}_atlas_flags.u8"))
print(f"wrote {TAG} atlas: {vocab} points, {int((flags == 2).sum())} clean words ({time.time() - t0:.0f}s)")
