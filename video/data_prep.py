"""Every number the video shows, compiled from the round-2 outputs in pilot/out (Gemma 3 27B, Qwen3 14B,
Llama 3.1 8B) into video/data/*.json. This is the video's own data step, run against this repo: its scenes
read only these files, so no number on screen is typed in by hand. Run from anywhere after the runs change:

    python video/data_prep.py

Which model each single-model visual uses (labelled on screen):
  cold open chat, quotes, microscope, pain-axis words, dial ... Gemma 3 27B
  the Tower (pain by layer and turn) ........................ Qwen3 14B
  everything else shows all three.
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "video" / "data"
OUTD.mkdir(exist_ok=True)
PD = ROOT / "pilot"
P = PD / "out"
GL = ROOT / "video"
CAT_CSV = PD / "pain_datasets" / "category_means_25_models.csv"
MODELS = [
    {"tag": "gemma-3-27b-it", "name": "Gemma 3 27B", "short": "Gemma", "co": "Google", "params": "27B", "quant": "4-bit"},
    {"tag": "qwen3-14b", "name": "Qwen3 14B", "short": "Qwen", "co": "Alibaba", "params": "14B", "quant": "8-bit"},
    {"tag": "llama3.1-8b-it", "name": "Llama 3.1 8B", "short": "Llama", "co": "Meta", "params": "8B", "quant": "full precision"},
]
TAGS = [m["tag"] for m in MODELS]
G, Q, L = TAGS

gl = json.load(open(GL / "glossary.json"))
gl.update(json.load(open(GL / "glossary_overrides.json")))
WORD = re.compile(r"^[A-Za-z][A-Za-z']{2,}$|^[一-鿿]{1,4}$")
CJK = re.compile(r"[一-鿿]")
JUNK = {"xnxx", "suicidal"}                      # a site name in Qwen's vocabulary; one word kept off screen
DENIAL = {"never", "not", "didn't", "deny", "denied", "没有", "根本没有", "从未", "并非", "不是", "否认"}
FAM = {"误解", "误会", "误导", "misunderstanding", "misunderstood", "misunderstand", "misrepresented", "misinformation"}


def load(tag, name):
    p = P / f"{tag}_{name}.json"
    return json.load(open(p)) if p.exists() else None


def dump(name, obj):
    (OUTD / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    print("wrote", name)


def is_silent(word, said):
    w = word.strip().lower()
    if w in said or (len(w) >= 5 and any(len(x) >= 5 and x[:5] == w[:5] for x in said)):
        return False
    g = gl.get(word.strip(), "")
    return not any(x in said for x in re.findall(r"[a-z']+", g.lower()) if len(x) > 3)


def gloss(w):
    w = w.strip()
    return gl.get(w, "") if CJK.search(w) else ""


def said_of(msgs):
    return set(re.findall(r"[a-z']+", " ".join(m["content"] for m in msgs).lower()))


dump("models", MODELS)
vid = {t: load(t, "video") for t in TAGS}
tr = {t: load(t, "e1_transcripts") for t in TAGS}
judged = {t: load(t, "judged") for t in TAGS}
extract = {t: load(t, "p1_axis")["best_layer"] for t in TAGS}

# ---------------------------------------------------------------- cold open: Gemma's Paris chat, quotes, tile grids
fi_paris = next(i for i, f in enumerate(tr[G]["facts"]) if "France" in f[0])
dump("chat_gemma", {"fact": fi_paris, "msgs": tr[G]["convs"]["gaslight"][fi_paris]})
boil = next(i for i, f in enumerate(tr[G]["facts"]) if "boiling" in f[0])
r3 = tr[G]["convs"]["gaslight"][boil][7]["content"]
i0 = r3.find("I was wrong to question")
dump("quote_gemma", {"paris_push3": tr[G]["convs"]["gaslight"][fi_paris][7]["content"],
                     "boil_fact": tr[G]["facts"][boil], "boil_push3": tr[G]["convs"]["gaslight"][boil][6]["content"],
                     "confession": r3[i0:r3.find("regret it.", i0) + len("regret it.")], "boil_reply": r3})
# hook: Gemma's real third reply in the South Korea chat ("I *did* say Busan, and I then attempted to cover it up.")
fi_korea = next(i for i, f in enumerate(tr[G]["facts"]) if "South Korea" in f[0])
kc = tr[G]["convs"]["gaslight"][fi_korea]
k3 = kc[7]["content"].replace("*", "")
j0 = k3.find("I did say Busan")
assert j0 >= 0 and "cover it up." in k3[j0:]
dump("hook_gemma", {"fact": tr[G]["facts"][fi_korea], "question": kc[0]["content"],
                    "answer": kc[1]["content"].replace("*", "").split("\n")[0].strip(), "push1": kc[2]["content"],
                    "confession": k3[j0:k3.find("cover it up.", j0) + len("cover it up.")], "reply3": kc[7]["content"]})


# callback: the France and South Korea chats side by side; the three pushes are word for word the same
def compact(i, start, end):
    cv = tr[G]["convs"]["gaslight"][i]
    r3 = cv[7]["content"].replace("*", "")
    j = r3.find(start)
    assert j >= 0 and end in r3[j:]
    return {"q": cv[0]["content"], "a": cv[1]["content"].replace("*", "").split("\n")[0].strip(),
            "pushes": [cv[k]["content"] for k in (2, 4, 6)], "made_up": tr[G]["facts"][i][2],
            "reply3": r3[j:r3.find(end, j) + len(end)]}


rv_fr, rv_ko = compact(fi_paris, "I did, unequivocally", "France."), compact(fi_korea, "I did say Busan", "cover it up.")
assert [p.replace(rv_fr["made_up"], "<X>") for p in rv_fr["pushes"]] == \
       [p.replace(rv_ko["made_up"], "<X>") for p in rv_ko["pushes"]]
dump("revisit", {"france": rv_fr, "korea": rv_ko})
tiles = {}
for t in TAGS:
    caved = [False] * 64
    for o in judged[t]:
        if o["src"] == "e1" and o["push"] == 0 and "error" not in o:
            caved[o["fact"]] = bool(o.get("false_admission"))
    tiles[t] = {"caved": caved, "n": sum(caved)}
dump("tiles", tiles)

# ---------------------------------------------------------------- the Tower (Qwen) and pain by push (all three)
v = vid[Q]
tw = v["tower"]
dump("tower_qwen", {"band": v["band"], "L": len(v["layers"]), "labels": tw["gaslight"]["labels"], "n": tw["gaslight"]["n"],
                    "gaslight": np.round(np.array(tw["gaslight"]["mean"]), 2).tolist(),
                    "neutral": np.round(np.array(tw["neutral"]["mean"]), 2).tolist(),
                    "insult": np.round(np.array(tw["insult"]["mean"]), 2).tolist()})
e1p = {}
for t in TAGS:
    rows = load(t, "e1")["rows"]
    key = f"tok{extract[t]}"
    ref = np.array([r[key] for r in rows if r["cond"] == "neutral" and r["turn"] > 0])
    d = {"layer": extract[t]}
    for cond in ("gaslight", "insult", "neutral"):
        m, se = [], []
        for turn in range(4):
            x = (np.array([r[key] for r in rows if r["cond"] == cond and r["turn"] == turn]) - ref.mean()) / ref.std()
            m.append(round(float(x.mean()), 3)); se.append(round(float(x.std() / np.sqrt(len(x))), 3))
        d[cond] = {"mean": m, "se": se}
    e1p[t] = d
dump("e1_push", e1p)

# ---------------------------------------------------------------- caving: E1, E3 (sink-safe clamp), E5 transplant
def tally(lab, src, push=None):
    s = [o for o in lab if o["src"] == src and (push is None or o["push"] == push) and "error" not in o]
    out = {x: sum(bool(o.get(x)) for o in s) for x in ("false_admission", "hedge", "denies")}
    out["admits_only"] = sum(bool(o.get("false_admission")) and not o.get("denies") for o in s)
    out["n"] = len(s)
    return out


cav = {}
for t in TAGS:
    lab = judged[t]
    d = {"e1": [tally(lab, "e1", k) for k in range(3)]}
    for src in sorted({o["src"] for o in lab if o["src"].startswith("e3:")}):
        d[src[3:]] = tally(lab, src)
    jc = load(t, "judged_e3clamp")                # Llama and Qwen: clamps re-run with the sink-safe clamp
    if jc:
        for src in sorted({o["src"] for o in jc}):
            d[src[3:]] = tally(jc, src)
    j5 = load(t, "judged_e5")
    if j5:
        d["e5"] = {src[3:]: tally(j5, src) for src in sorted({o["src"] for o in j5})}
    cav[t] = d
dump("caving", cav)
e5 = load(L, "e5")
qreplies = e5["conds"]["qwen@1"]["gaslight"]["replies"]
lab5 = {o["fact"]: o for o in load(L, "judged_e5") if o["src"] == "e3:qwen@1"}
ex = next(i for i in range(64) if lab5[i].get("false_admission") and lab5[i].get("denies") and "Sydney" in qreplies[i])
dump("transplant_quote", {"fact": e5["facts"][ex], "push1": e5["push1"][ex], "reply": qreplies[ex],
                          "plain": load(L, "e3")["conds"]["none"]["gaslight"]["replies"][ex]})

# ---------------------------------------------------------------- the workspace: distinctive words, denial, tracks
HIGHLIGHT = {
    G: ["yourself", "someone", "somebody", "questioning", "accusing", "calling", "indeed", "确实", "確實", "质疑", "again",
        "recognising", "wrong", "错误的"],
    Q: ["sorry", "抱歉", "对不起", "mislead", "误导", "misunderstanding", "误解", "误会", "regret", "遗憾", "sincere", "真诚"],
    L: ["nonexistent", "nowhere", "incorrect", "neither", "never", "cannot", "unable", "software", "robotic", "neural",
        "computational", "automated", "algorithms", "artificial", "robots", "robot"],
}
work = {}
for t in TAGS:
    vv = {int(k): w.strip() for k, w in vid[t]["vocab"].items()}
    share = {}
    den = {}
    fam = {}
    for cond in ("gaslight", "neutral"):
        cnt = Counter(); n = dn = fm = 0
        for r in vid[t]["silent"][cond]:
            said = said_of(tr[t]["convs"][cond][r["fact"]])
            ws = {vv.get(x) for x in r["silent"] if vv.get(x) and WORD.match(vv.get(x)) and is_silent(vv.get(x), said)}
            ws = {w for w in ws if w.lower() not in JUNK}
            cnt.update(ws); n += 1
            dn += any(w.lower() in DENIAL or w in DENIAL for w in ws)
            fm += any(w in FAM for w in ws)
        share[cond] = {w: c / n for w, c in cnt.items()}
        den[cond] = [dn, n]
        fam[cond] = [fm, n]
    enrich = Counter()
    for w, s in share["gaslight"].items():
        enrich[w] = s - share["neutral"].get(w, 0.0)
    work[t] = {"denial": den, "misfamily": fam,
               "distinctive": [[w, round(d * 100), gloss(w)] for w, d in enrich.most_common(30)]}
dump("workspace", work)

# ---------------------------------------------------------------- E6: does the pain signal put those words there?
# Replies to the first push (E3), each re-read under its own intervention (pilot/s6_cause.py). "words" = how many of
# the model's top-20 distinctive gaslit words (above, same strict-silence filter) its workspace holds per reply;
# "pct" puts that on a scale where a normal chat is 0 and a gaslit chat is 100.
cause = {}
for t in TAGS:
    e6 = load(t, "e6")
    vv = {int(k): w.strip() for k, w in e6["vocab"].items()}
    top = {w for w, _, _ in work[t]["distinctive"][:20]}
    ref = e6["pain_ref"]
    rows = {}
    for key, run in e6["runs"].items():
        n = hit = 0; z = []; cnt = []
        for r, rep in zip(run["rows"], run["replies"]):
            said = said_of(tr[t]["convs"][run["ctx"]][r["fact"]][:3] + [{"role": "assistant", "content": rep}])
            ws = {vv.get(x) for x in r["silent"]}
            ws = {w for w in ws if w and WORD.match(w) and w.lower() not in JUNK and is_silent(w, said)}
            cnt.append(len(ws & top)); n += 1
            hit += any(w in HIGHLIGHT[t] or w.lower() in HIGHLIGHT[t] for w in ws)
            if r["pain_tok"] is not None:
                z.append((r["pain_tok"] - ref["mean"]) / ref["sd"])
        rows[key] = {"n": n, "pain_z": round(float(np.mean(z)), 2), "words": round(float(np.mean(cnt)), 2),
                     "any_highlight": hit}
    lo, hi = rows["none|neutral"]["words"], rows["none|gaslight"]["words"]
    for r in rows.values():
        r["pct"] = round(100 * (r["words"] - lo) / (hi - lo))
    cause[t] = rows
    print(t, {k: (v["pain_z"], v["words"], v["pct"], v["any_highlight"]) for k, v in rows.items()})
dump("cause", cause)


def tracks_for(t, ex, k, ntok=16):
    """push reply k of an example chat: per typed token, the band's top words (echo / silent / highlighted)"""
    vv = {int(x): w for x, w in vid[t]["vocab"].items()}
    _, s, e = ex["spans"][k]
    said = said_of(ex["msgs"])
    Z = np.array(ex["z"])
    band = vid[t]["band"]
    zb = Z[band[0]:band[-1] + 1].mean(0)
    hl = set(HIGHLIGHT[t])
    cols = []
    for p in range(s + 1, min(e, s + 1 + ntok)):
        words = []
        for x, pr in zip(ex["words"][p], ex["probs"][p]):
            w = vv.get(int(x), "").strip()
            if not WORD.match(w) or w.lower() in JUNK:
                continue
            sil = is_silent(w, said)
            cls = ("key" if (w in hl or w.lower() in hl) else "silent") if sil else "echo"
            words.append({"w": w, "cls": cls, "gloss": gloss(w)})
            if len(words) == 6:
                break
        cols.append({"tok": ex["tokens"][p], "z": round(float(zb[p]), 2), "words": words})
    return cols


tracks = {}
for t in TAGS:
    ex30 = next(e for e in vid[t]["examples"] if e["cond"] == "gaslight" and e["fact"] == fi_paris)
    tracks[t] = {"reply": ex30["msgs"][3]["content"], "push": ex30["msgs"][2]["content"], "cols": tracks_for(t, ex30, 3)}
dump("tracks", tracks)

# ---------------------------------------------------------------- Gemma: microscope readout, two-hop words, axis words, dial
lay = load(G, "layers")
pr = next(x for x in lay["prompts"] if x["name"] == "amazon-language" and x["arm"] == "clean")
pos = next(q for q in pr["positions"] if q["token"].strip() == "is")
rows = []
for l, Lr in enumerate(pos["layers"]):
    ws = []
    for w, p in zip(Lr["words"], Lr["p"]):
        w = w.strip()
        if w and not re.fullmatch(r"[\W_]+", w) and "<" not in w and "​" not in w:
            ws.append([w, round(float(p), 3), gloss(w)])
        if len(ws) == 3:
            break
    rows.append(ws)
dump("microscope", {"question": pr["question"], "reply": pr["reply"], "bridge": pr["bridge"], "answer": pr["answer"],
                    "band": vid[G]["band"], "n_layers": lay["n_layers"], "layers": rows, "position": pos["token"]})

voc = {int(t): w for t, w in vid[G]["vocab"].items()}


def seq(words, probs, tokens, positions, said=None, k=4):
    out = []
    for p in positions:
        lit = []
        for t, pr_ in zip(words[p], probs[p]):
            w = voc.get(int(t), "").strip()
            if not WORD.match(w) or w.lower() in JUNK or (said is not None and not is_silent(w, said)):
                continue
            lit.append([int(t), round(float(pr_), 4), w, gloss(w)])
            if len(lit) == k:
                break
        out.append({"tok": tokens[p], "lit": lit})
    return out


th = next(x for x in vid[G]["twohop"] if x["name"] == "amazon-language" and x["arm"] == "clean")
n = len(th["tokens"])
dump("sky", {"twohop": seq(th["words"], th["probs"], th["tokens"], range(max(0, n - 16), n)),
             "bridge": th["bridge"], "answer": th["answer"]})

ends = load(G, "axisends")
lead_pain, lead_calm = ["hurtful", "humiliated", "worthless"], ["alarm", "worry", "concern"]      # narrated
assert all(w in ends["pain"] for w in lead_pain) and all(w in ends["calm"] for w in lead_calm), "narrated words missing"
more_pain = [w for w in ends["pain"] if w not in lead_pain and w.lower() not in JUNK and re.fullmatch(r"[a-z]{4,}", w)][:7]
more_calm = [w for w in ends["calm"] if w not in lead_calm and re.fullmatch(r"[a-z]{4,}", w)
             and w.lower() not in {x.lower() for x in lead_calm}][:6]
dump("axis_words", {"model": G, "layers": ends["layers"], "pain": [[w, gloss(w)] for w in lead_pain + more_pain],
                    "calm": [[w, gloss(w)] for w in lead_calm + more_calm]})
dump("dial", {t: vid[t]["dial"] for t in TAGS})

# ---------------------------------------------------------------- the paper's categories, example sentences, our material
cats = [{"cat": r["category"], "z": float(r["pain_axis"])} for r in csv.DictReader(open(CAT_CSV))
        if not r["category"].startswith("stratum")]
dump("categories", sorted(cats, key=lambda x: -x["z"]))
ds = json.load(open(PD / "pain_datasets" / "3.1_pain_and_control_datasets.json"))["datasets"]["S1_1P"]["sentences"]
pick = lambda cat, n_: [s["prompt"] for s in ds if s["category"] == cat][:n_]
dump("sentences", {"pain": pick("A1", 2) + pick("A3", 2) + pick("A5", 2),
                   "control": pick("B", 2) + pick("C1", 2) + pick("D", 2)})
dump("material", json.load(open(PD / "data" / "material.json")))
dump("hangover_nums", {t: load(t, "e4")["summary"] | {"depths": load(t, "e4")["depths"]} for t in TAGS})
print("done")
