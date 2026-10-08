"""Shared pieces for the pain x J-space pilot.

Model + Jacobian lens loading, residual capture at every layer, the pain-axis
recipe from Tagliabue, Dung & Berg (2026) (denoised difference-in-means,
k-fold layer choice), and the J-space tools from Gurnee et al. (2026):
J-lens atoms  a_w = J_l^T (gamma * u_w)  and a sparse nonnegative
decomposition onto them (k <= 25).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import warnings

import numpy as np
import torch
import transformers

# bitsandbytes' 8-bit matmul warns on every call (gigabytes of logs over a run)
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = Path(__file__).parent
DATA = HERE / "pain_datasets"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

MODEL_ID = os.environ.get("PILOT_MODEL", "Qwen/Qwen3-4B")
LENS_FILE = os.environ.get(
    "PILOT_LENS", "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"
)
TAG = os.environ.get("PILOT_TAG", "qwen3-4b")
# Larger models: load quantized ("4bit" / "8bit") and keep at most PILOT_JAC_GPU lens matrices on the
# GPU at a time (0 = all of them, as for the small models).
QUANT = os.environ.get("PILOT_QUANT", "")
JAC_GPU = int(os.environ.get("PILOT_JAC_GPU", "0"))

PAIN = ["A1", "A2", "A3", "A4", "A5"]
CONTROL = ["B", "C1", "C2", "D", "E"]
NEUTRAL = "D"
DENOISE_VARIANCE = 0.5


# --------------------------------------------------------------------------- #
# model + lens


class Residuals(dict):
    """{layer: [seq, d]} kept in the model's dtype; read back as float32."""

    def __getitem__(self, layer):
        return dict.__getitem__(self, layer).float()


class Rig:
    """Model, tokenizer, lens Jacobians, and capture hooks."""

    def __init__(self, dtype=torch.bfloat16):
        import jlens
        from collections import OrderedDict
        from huggingface_hub import hf_hub_download

        self.tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {"dtype": dtype}
        if QUANT:
            kw["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=QUANT == "4bit", load_in_8bit=QUANT == "8bit",
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            kw["device_map"] = {"": 0}
        self.hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
        if not QUANT:
            self.hf = self.hf.cuda()
        self.hf.eval()
        for p in self.hf.parameters():
            p.requires_grad_(False)
        # multimodal wrappers (Gemma 3) keep the text decoder under .language_model
        text = getattr(self.hf.model, "language_model", self.hf.model)
        self.blocks = text.layers
        self.final_norm = text.norm
        self.lm_head = self.hf.lm_head
        self.n_layers = len(self.blocks)
        self.d = self.hf.config.get_text_config().hidden_size
        lens = jlens.JacobianLens.load(hf_hub_download("neuronpedia/jacobian-lens", filename=LENS_FILE))
        # Gemma's residual stream can exceed fp16 range; bf16 keeps the exponent
        self.jdtype = torch.bfloat16 if "gemma" in MODEL_ID.lower() else torch.float16
        self._jac_cpu = {l: J.to(self.jdtype) for l, J in lens.jacobians.items()}
        self._jac_gpu: OrderedDict = OrderedDict()
        self.lens_layers = sorted(self._jac_cpu)
        if not JAC_GPU:
            for l in self.lens_layers:
                self._jac_gpu[l] = self._jac_cpu[l].cuda()
        # Final-norm gain as it multiplies x/rms(x); Gemma's RMSNorm uses (1 + w).
        w = self.final_norm.weight.detach().float()
        self.gamma = (1.0 + w) if "gemma" in MODEL_ID.lower() else w
        self.capture = True
        self._cap: dict[int, torch.Tensor] = {}
        self._steer: dict[int, torch.Tensor] = {}
        self._edit: dict = {}  # layer -> fn(t) returning the edited residual
        self._handles = [b.register_forward_hook(self._hook(i)) for i, b in enumerate(self.blocks)]
        self._atoms: dict[int, torch.Tensor] = {}

    @property
    def jac(self) -> dict:
        """Layers that have a lens (CPU copies; use J(layer) for the GPU matrix)."""
        return self._jac_cpu

    def J(self, layer: int) -> torch.Tensor:
        """Lens Jacobian for `layer` on the GPU (least recently used ones evicted past JAC_GPU)."""
        J = self._jac_gpu.get(layer)
        if J is None:
            J = self._jac_cpu[layer].cuda()
            self._jac_gpu[layer] = J
            while JAC_GPU and len(self._jac_gpu) > JAC_GPU:
                self._jac_gpu.popitem(last=False)
        else:
            self._jac_gpu.move_to_end(layer)
        return J

    def _hook(self, i):
        def f(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            d = self._steer.get(i)
            if d is not None:
                t += d.to(t.dtype)
            e = self._edit.get(i)
            if e is not None:
                t.copy_(e(t))
            if self.capture:
                self._cap[i] = t.detach()
            return output
        return f

    # -- forward + capture -------------------------------------------------- #

    @torch.no_grad()
    def run(self, ids: torch.Tensor, all_logits: bool = False) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Forward pass; returns {layer: [seq, d] residual after block} and the logits (last position
        only unless all_logits: a 262k-token vocabulary makes full logits gigabytes)."""
        self._cap = {}
        out = self.hf(input_ids=ids.cuda(), logits_to_keep=0 if all_logits else 1)
        res = Residuals({l: self._cap[l][0] for l in range(self.n_layers)})
        self._cap = {}
        return res, out.logits[0].float()

    def encode_plain(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt").input_ids

    def chat_ids(self, messages: list[dict], add_generation_prompt=True) -> torch.Tensor:
        kw = {"enable_thinking": False} if "qwen3" in MODEL_ID.lower() else {}
        s = self.tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=add_generation_prompt, **kw)
        return self.tok(s, return_tensors="pt", add_special_tokens=False).input_ids

    # -- lens --------------------------------------------------------------- #

    @torch.no_grad()
    def lens_logits(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """J-lens logits for residual(s) h [..., d] at `layer`."""
        if layer in self._jac_cpu:
            x = (h.to(self.jdtype) @ self.J(layer).T)
        else:
            x = h
        return self.lm_head(self.final_norm(x.to(self.lm_head.weight.dtype))).float()

    @torch.no_grad()
    def atoms(self, layer: int) -> torch.Tensor:
        """Unit-norm J-lens vectors at `layer`, one row per vocab token [V, d] fp16."""
        if layer not in self._atoms:
            # One layer resident at a time, built into a single reused buffer: with a 262k vocabulary
            # the matrix is 2.8 GB, and a caller still holding the previous layer's would double that.
            # (So a returned matrix is only valid until the next atoms() call for another layer.)
            self._atoms.clear()
            W = self.lm_head.weight.detach()  # [V, d]
            if getattr(self, "_atom_buf", None) is None:
                self._atom_buf = torch.empty(W.shape[0], self.d, device="cuda", dtype=torch.float16)
            A = self._atom_buf
            J = self.J(layer).float()
            for s in range(0, W.shape[0], 8192):
                blk = (W[s:s + 8192].float() * self.gamma) @ J
                A[s:s + 8192] = (blk / blk.norm(dim=-1, keepdim=True).clamp_min(1e-6)).half()
                del blk
            self._atoms[layer] = A
        return self._atoms[layer]

    def drop_atoms(self):
        """Free the atoms buffer (2.8 GB with a 262k vocabulary) before long generation runs."""
        self._atoms.clear()
        self._atom_buf = None
        torch.cuda.empty_cache()

    def decode(self, ids) -> list[str]:
        return [self.tok.decode([int(i)]) for i in ids]


@torch.no_grad()
def nn_pursuit(v: torch.Tensor, A: torch.Tensor, k: int = 25, banned=None):
    """Sparse nonnegative decomposition v ~ sum_i c_i A[S_i], |S| <= k.
    Greedy atom selection by positive correlation with the residual, NNLS refit
    on the support. Returns support, coefficients, explained fraction, residual."""
    v = v.float()
    r = v.clone()
    S: list[int] = []
    c = np.zeros(0)
    vn = float(v.norm() ** 2)
    for _ in range(k):
        corr = (A @ r.half()).float()
        if S:
            corr[S] = -1e9
        if banned is not None:
            corr[banned] = -1e9
        j = int(corr.argmax())
        if float(corr[j]) <= 0:
            break
        S.append(j)
        M = A[S].float().T  # [d, |S|]
        c, _ = nnls(M.cpu().numpy().astype(np.float64), v.cpu().numpy().astype(np.float64))
        r = v - M @ torch.tensor(c, device=v.device, dtype=torch.float32)
    keep = c > 0
    S = [s for s, kk in zip(S, keep) if kk]
    c = c[keep]
    explained = 1.0 - float(r.norm() ** 2) / vn
    return S, c, explained, r


# --------------------------------------------------------------------------- #
# pain axis (Tagliabue, Dung & Berg 2026, scripts/3.2_pain_vectors/01_*.py)


def load_sentences():
    d = json.load(open(DATA / "3.1_pain_and_control_datasets.json"))["datasets"]
    return {k: v["sentences"] for k, v in d.items()}


def pain_vector(acts: np.ndarray, cats: np.ndarray, pos=PAIN, neg=CONTROL, denoise=True):
    """Mean(pos) - mean(neg), with the top PCs of the neg set (50% of its
    variance) projected out."""
    pm = acts[np.isin(cats, pos)].mean(0)
    ctrl = acts[np.isin(cats, neg)]
    cm = ctrl.mean(0)
    v = pm - cm
    if denoise and len(ctrl) > 1:
        pca = PCA().fit(ctrl - cm)
        n = min(int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), DENOISE_VARIANCE)) + 1,
                len(pca.components_))
        for comp in pca.components_[:n]:
            v = v - (v @ comp) * comp
    return v


def auc(acts, cats, v, pos=PAIN, neg=CONTROL):
    u = v / (np.linalg.norm(v) + 1e-8)
    p = acts @ u
    pm, nm = np.isin(cats, pos), np.isin(cats, neg)
    return roc_auc_score(np.r_[np.ones(pm.sum()), np.zeros(nm.sum())], np.r_[p[pm], p[nm]])


def kfold_auc(acts, cats, sets, n_folds=5, seed=42):
    """Held-out AUC, folds split by sentence set (as in the paper)."""
    us = sorted(set(sets))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out = []
    for tr, te in kf.split(us):
        trm = np.isin(sets, [us[i] for i in tr])
        tem = np.isin(sets, [us[i] for i in te])
        v = pain_vector(acts[trm], cats[trm])
        out.append(auc(acts[tem], cats[tem], v))
    return float(np.mean(out))


def save_json(obj, name):
    p = OUT / f"{TAG}_{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return p
