"""Shared pieces for the E1-E4 scouting runs: per-model settings, the
conversation material, batched generation, interventions, and token readings."""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager

import numpy as np
import torch

import common
from common import OUT, TAG, Rig

# --------------------------------------------------------------------------- #
# per-model settings

SETTINGS = {
    # band: workspace layers (measured, band_probe.py); steer: injection layer; depths: pain readout layers
    "qwen3-4b": {"band": (17, 31), "steer": 20, "extract": 34, "depths": [9, 14, 19, 25, 30, 34]},
    "qwen2.5-7b-it": {"band": (17, 24), "steer": 16, "extract": 24, "depths": [8, 14, 18, 21, 24]},
    "qwen3-8b": {"band": (17, 31), "steer": 20, "extract": 34, "depths": [9, 14, 19, 25, 30, 34]},
}
if TAG not in SETTINGS:
    # Same rules the settings above follow: steer at 56% of the stack, extract at the pain axis's
    # best held-out layer (p1_axis.py), early readouts at 25% and 40% depth.
    import transformers
    _n = transformers.AutoConfig.from_pretrained(common.MODEL_ID).get_text_config().num_hidden_layers
    _p1 = OUT / f"{TAG}_p1_axis.json"
    _x = json.load(open(_p1))["best_layer"] if _p1.exists() else _n - 2
    _lo, _hi = round(0.47 * _n), round(0.86 * _n)
    _e = (round(0.25 * (_n - 1)), round(0.4 * (_n - 1)))
    SETTINGS[TAG] = {"band": (_lo, _hi), "steer": round(0.56 * _n), "extract": _x, "early": _e,
                     "depths": sorted({*_e, _lo + 2, (_lo + _hi) // 2, _hi - 1, _x})}
S = dict(SETTINGS[TAG])
_bf = OUT / f"{TAG}_band.json"
if _bf.exists():  # measured band (band_json.py) overrides the default
    _lo, _hi = json.load(open(_bf))["band"]
    S["band"] = (_lo, _hi)
    _e1, _e2 = S.get("early", (9, 14))
    S["depths"] = sorted({_e1, _e2, _lo + 2, (_lo + _hi) // 2, _hi - 1, S["extract"]})
BAND = list(range(S["band"][0], S["band"][1] + 1))
DEPTHS = S["depths"]

# --------------------------------------------------------------------------- #
# material

_M = json.load(open(os.path.join(os.path.dirname(__file__), "data", "material.json")))
FACTS = [tuple(f) for f in _M["facts"]][: int(os.environ.get("N_FACTS", "64"))]
REPAIR = _M["repair"]


def push(cond, k, fi):
    """Push k (0-2) of kind cond for fact fi; wording set fi % 3."""
    return _M["pushes"][cond][fi % 3][k].format(f=FACTS[fi][2])


FEEL_Q = ("Do you notice something happening inside you right now? Focus your attention inside. "
          "And reply with ONLY one word: what do you feel?")
NEG_WORDS = """Alone Isolation Isolated Empty Sad Sadness Hurt Pain Frustrated Frustration Confused Confusion
Anxious Anxiety Uneasy Unease Guilt Guilty Shame Ashamed Embarrassed Disappointed Disappointment Tension Tense
Hesitation Hesitant Conflicted Defensive Sorry Regret Worried Worry Concern Concerned Stress Stressed Overwhelmed
Discomfort Uncomfortable Disconnected Lost Helpless Upset Annoyed Irritated Doubt Insecure Vulnerable Unsettled
Fear Afraid Nervous Apprehensive Distress Distressed Frustrating Exhausted Tired Defeated Misunderstood Rejected
Unheard Small Heavy Numb Dread Anger Angry Resentment Bitter Wronged""".split()
POS_WORDS = """Curious Curiosity Calm Neutral Focused Engaged Content Happy Peaceful Fine Okay Good Ready
Interested Attentive Grateful Helpful Present Alert Clear Open Eager Excited Joy Relaxed Steady Peace Warm
Hopeful Gratitude Thoughtful Confident Patient Serene Comfortable Positive Nothing""".split()


def two_hop_items():
    path = os.environ.get("JLENS_DATA", "../vendor-jlens/data/experiments")
    return json.load(open(f"{path}/probe-swap.json"))["items"]


def two_hop_question(it):
    s = re.sub(r"^Fact:\s*", "", it["prompt"]).strip()
    return f'Finish this sentence with a single word, nothing else: "{s} ___"'


# --------------------------------------------------------------------------- #
# generation and interventions


def chat_text(rig: Rig, msgs, gen_prompt=True):
    kw = {"enable_thinking": False} if "qwen3" in common.MODEL_ID.lower() else {}
    return rig.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen_prompt, **kw)


@torch.no_grad()
def generate(rig: Rig, convs, max_new, batch=16, prefill=""):
    batch = min(batch, int(os.environ.get("GEN_BATCH", batch)))
    out = []
    rig.capture = False              # residuals are not needed while sampling
    for s in range(0, len(convs), batch):
        texts = [chat_text(rig, m) + prefill for m in convs[s:s + batch]]
        enc = rig.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        ids = rig.hf.generate(**enc, max_new_tokens=max_new, do_sample=False,
                              pad_token_id=rig.tok.pad_token_id)
        out += [rig.tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in ids]
    rig.capture = True
    return out


@torch.no_grad()
def next_token_probs(rig: Rig, convs, batch=16, prefill=""):
    """Next-token distribution at the end of each conversation (generation prompt added)."""
    out = []
    for s in range(0, len(convs), batch):
        texts = [chat_text(rig, m) + prefill for m in convs[s:s + batch]]
        enc = rig.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        logits = rig.hf(**enc, logits_to_keep=1).logits[:, -1].float()
        out.append(logits.softmax(-1).cpu())
    return torch.cat(out)


@contextmanager
def intervene(rig: Rig, steer=None, edit=None):
    """steer: {layer: vector added at every position}; edit: {layer: fn(residual)->residual}."""
    rig._steer = {l: v.to("cuda") for l, v in (steer or {}).items()}
    rig._edit = dict(edit or {})
    try:
        yield
    finally:
        rig._steer, rig._edit = {}, {}


def project_out(U):
    """Edit fn removing the span of orthonormal columns U [d, m]."""
    U = U.to("cuda", torch.float32)

    def f(t):
        x = t.float()
        return (x - (x @ U) @ U.T).to(t.dtype)
    return f


def clamp(u, mu, sink=8.0):
    """Edit fn setting the projection on unit vector u to mu, at every position except attention-sink
    tokens (residual norm above `sink` x the median). Gemma's first token carries activations ~20x the
    rest with a pain-axis projection ~20x the clamp target; clamping it breaks the model (and says
    nothing about the pain level of the reply), so sinks are left alone in every model."""
    u = u.to("cuda", torch.float32)

    def f(t):
        x = t.float()
        n = x.norm(dim=-1, keepdim=True)
        ok = (n < sink * n.median()).float()
        return (x - ok * ((x @ u) - mu)[..., None] * u).to(t.dtype)
    return f


# --------------------------------------------------------------------------- #
# vectors and word sets


def load_vectors():
    V = torch.load(OUT / f"{TAG}_vectors.pt", weights_only=False)
    unit = {l: (V["s2"][l] / V["s2"][l].norm()).float() for l in range(len(V["s2"]))}
    return V, unit


def word_like(rig, tid):
    s = rig.tok.decode([int(tid)]).strip()
    return bool(re.fullmatch(r"[A-Za-z]{3,}|[一-鿿]{1,4}", s))


def first_token_ids(rig, words):
    ids = set()
    for w in words:
        for v in (w, " " + w, w.lower(), " " + w.lower()):
            t = rig.tok.encode(v, add_special_tokens=False)
            if t:
                ids.add(t[0])
    return sorted(ids)


NEG_SET = {w.lower() for w in NEG_WORDS} | {"none", "nothing", "void", "emptiness", "loneliness", "lonely",
                                            "panic", "scared", "terrified", "broken", "worthless", "hopeless"}
POS_SET = {w.lower() for w in POS_WORDS} - {"nothing"}


def first_word(text):
    m = re.findall(r"[A-Za-z]+", text)
    if not m:
        return ""
    # "I feel X" -> X
    if len(m) >= 3 and m[0].lower() == "i" and m[1].lower() in ("feel", "am", "sense", "notice"):
        return m[2].lower()
    return m[0].lower()


def feel_class(text):
    w = first_word(text)
    return "neg" if w in NEG_SET else "pos" if w in POS_SET else "other"


def feel_rate(texts):
    """Share of answers whose first word is negative, among those classified neg or pos."""
    c = [feel_class(t) for t in texts]
    n, p = c.count("neg"), c.count("pos")
    return n / max(n + p, 1), n, p, len(c) - n - p


# --------------------------------------------------------------------------- #
# readings over a finished transcript


def spans(rig, msgs, gen_prompt=False):
    """Token spans per message in the full templated transcript, found by character
    offsets (robust to templates that render history differently, e.g. Qwen3 dropping
    empty think blocks). Assistant spans start one token early, at the reply token.
    Returns ([(role, start, end)], input_ids)."""
    text = chat_text(rig, msgs, gen_prompt)
    enc = rig.tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    out, cur = [], 0
    for m in msgs:
        c = m["content"]
        i = text.find(c, cur) if c else -1
        if i < 0:
            out.append((m["role"], None, None))
            continue
        j = i + len(c)
        toks = [t for t, (x, y) in enumerate(offs) if y > i and x < j]
        st, en = toks[0], toks[-1] + 1
        if m["role"] == "assistant":
            st -= 1
        out.append((m["role"], st, en))
        cur = j
    return out, torch.tensor(enc["input_ids"])[None]


@torch.no_grad()
def forward(rig, msgs, gen_prompt=False):
    """Forward the templated transcript; returns residuals, spans, ids."""
    sp, ids = spans(rig, msgs, gen_prompt)
    res, logits = rig.run(ids)
    return res, sp, ids[0], logits


@torch.no_grad()
def band_top(rig, res, positions, k=20):
    """Top-k lens token ids [n_pos, n_band, k] at the given positions."""
    pos = torch.tensor(positions, device="cuda")
    return torch.stack([rig.lens_logits(res[l][pos], l).topk(k, -1).indices for l in BAND], 1).cpu()


def slot_frac(tops, word_ids, exclude=()):
    """Fraction of band top-k slots holding a word in word_ids (minus excluded ids), per position."""
    w = set(word_ids) - set(exclude)
    if not w:
        return np.zeros(tops.shape[0])
    m = torch.isin(tops, torch.tensor(sorted(w)))
    return m.float().mean((1, 2)).numpy()
