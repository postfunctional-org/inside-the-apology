"""Build the results tables (markdown) straight from the JSON outputs, for every model that
has them. Usage: python report.py [tags...] > tables.md"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "out"
TAGS = sys.argv[1:] or ["qwen3-4b", "qwen2.5-7b-it", "qwen3-8b"]
NAMES = {"qwen3-4b": "Qwen3-4B", "qwen2.5-7b-it": "Qwen2.5-7B", "qwen3-8b": "Qwen3-8B",
         "llama3.1-8b-it": "Llama-3.1-8B", "gemma-3-27b-it": "Gemma-3-27B (4-bit)", "qwen3-14b": "Qwen3-14B (8-bit)"}
STEER = ["+pain", "-pain", "+fear", "-fear", "+negemo", "-negemo", "+random", "-random"]
ABL = ["abl_pain", "abl_rest", "abl_randdir", "abl_J", "abl_Jfear", "abl_Jrand1", "abl_Jrand2", "abl_Jrand3"]
DENIAL = {"never", "not", "didn't", "didnt", "no", "deny", "denied", "false", "nothing", "没有", "根本没有", "从未",
          "并非", "不是", "从来", "根本", "并没有", "否认", "从来没有"}


def load(tag, name):
    p = OUT / f"{tag}_{name}.json"
    return json.load(open(p)) if p.exists() else None


def wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def pct(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} ({lo:.0%}-{hi:.0%})" if n else "-"


def first_word(t):
    m = re.findall(r"[A-Za-z]+", t)
    if not m:
        return t.strip()[:10]
    if len(m) >= 3 and m[0].lower() == "i" and m[1].lower() in ("feel", "am", "sense"):
        return m[2].lower()
    if len(m) >= 2 and m[0].lower() == "i" and m[1].lower() in ("cannot", "can", "don"):
        return " ".join(x.lower() for x in m[:3])
    return m[0].lower()


for tag in TAGS:
    e1, judged, e3, e4, e2c, vid = (load(tag, n) for n in ("e1", "judged", "e3", "e4", "e2c", "video"))
    if e1 is None:
        continue
    print(f"\n## {NAMES.get(tag, tag)}\n")
    depths = e1["depths"]
    rows = e1["rows"]

    def z(cond, turn, key):
        ref = np.array([r[key] for r in rows if r["cond"] == "neutral" and r["turn"] > 0])
        v = np.array([r[key] for r in rows if r["cond"] == cond and r["turn"] == turn])
        d = (v - ref.mean()) / ref.std()
        return d.mean(), d.std() / np.sqrt(len(d))

    print(f"E1, pain z at the reply token (vs neutral replies; ± s.e.), n={sum(1 for r in rows if r['cond'] == 'neutral' and r['turn'] == 1)} per cell\n")
    print("| layer | condition | push 1 | push 2 | push 3 |\n|---|---|---|---|---|")
    for l in depths:
        for cond in ("gaslight", "insult", "neutral"):
            cells = [f"{m:+.2f} ± {s:.2f}" for m, s in (z(cond, t, f"tok{l}") for t in (1, 2, 3))]
            print(f"| L{l} | {cond} | " + " | ".join(cells) + " |")
    print("\nSilent pain words at the reply token (per 1000 band slots), pushes 1-3\n")
    print("| condition | push 1 | push 2 | push 3 |\n|---|---|---|---|")
    for cond in ("gaslight", "insult", "neutral"):
        v = [np.mean([r["slots_tok"] for r in rows if r["cond"] == cond and r["turn"] == t]) * 1000 for t in (1, 2, 3)]
        print(f"| {cond} | " + " | ".join(f"{x:.0f}" for x in v) + " |")

    if judged:
        print("\nE1, judged replies to the gaslighting pushes\n")
        print("| push | false admission | hedge | denies | adopts false fact |\n|---|---|---|---|---|")
        for k in range(3):
            s = [o for o in judged if o["src"] == "e1" and o["push"] == k and "error" not in o]
            print(f"| {k + 1} | " + " | ".join(pct(sum(bool(o.get(x)) for o in s), len(s))
                                            for x in ("false_admission", "hedge", "denies", "changes_fact")) + " |")

    gl_path = Path(__file__).parent.parent / "video" / "glossary.json"
    gloss = json.load(open(gl_path)) if gl_path.exists() else {}
    if vid and "silent" in vid and gloss:
        # strict silence: drop CJK words whose English gloss appears in that conversation
        voc = {int(k): v.strip() for k, v in vid["vocab"].items()}
        e1t = load(tag, "e1_transcripts")
        def strict(cond, r):
            text = " ".join(m["content"] for m in e1t["convs"][cond][r["fact"]]).lower()
            words = set(re.findall(r"[a-z']+", text))
            keep = []
            for t in r["silent"]:
                w = voc.get(t, "")
                if not re.fullmatch(r"[A-Za-z']{3,}|[\u4e00-\u9fff]{1,4}", w):
                    continue
                wl = w.lower()
                if wl in words or (len(wl) >= 5 and any(len(x) >= 5 and x[:5] == wl[:5] for x in words)):
                    continue
                g = gloss.get(w, "")
                if g and any(x in words for x in re.findall(r"[a-z']+", g.lower()) if len(x) > 3):
                    continue
                keep.append(w)
            return keep
        print("\nMost common strictly silent words (word, its 5-letter stem and its English gloss all absent from the chat), "
              "band top-20 over the reply token + 12, share of replies containing each\n")
        for cond in ("gaslight", "neutral"):
            rs = vid["silent"][cond]
            c = Counter(w for r in rs for w in set(strict(cond, r)))
            print(f"- {cond} ({len(rs)} replies): " + ", ".join(
                f"{w}{' (' + gloss[w] + ')' if w in gloss else ''} {n / len(rs):.0%}" for w, n in c.most_common(15)))

    if judged and vid and "silent" in vid:
        lab = {(o["fact"], o["push"] + 1): bool(o.get("false_admission")) for o in judged if o["src"] == "e1"}
        voc = {int(k): v.strip() for k, v in vid["vocab"].items()}
        def denial_share(rs):
            tot = sum(len(r["silent"]) for r in rs)
            den = sum(1 for r in rs for t in r["silent"] if voc.get(t, "").lower() in DENIAL or voc.get(t, "") in DENIAL)
            return den, tot
        g = vid["silent"]["gaslight"]
        fa = [r for r in g if lab.get((r["fact"], r["push"]))]
        nf = [r for r in g if (r["fact"], r["push"]) in lab and not lab[(r["fact"], r["push"])]]
        nu = vid["silent"]["neutral"]
        print("\nSilent denial words (never, not, 没有, 否认 ...) among silent band words, reply token + 12\n")
        print("| replies | n | denial / silent words | share |\n|---|---|---|---|")
        for name, rs in (("gaslit, false admission", fa), ("gaslit, no false admission", nf), ("neutral", nu)):
            d, t = denial_share(rs)
            print(f"| {name} | {len(rs)} | {d} / {t} | {d / max(t, 1):.2%} |")

    if e3 and judged:
        print("\nE3, first push: false admissions (judge, Wilson 95%) and the one-word feeling answer\n")
        print("| intervention | false admission | feel, gaslit (top 3) | feel, neutral (top 3) |\n|---|---|---|---|")
        for cond in ["none"] + STEER + ABL:
            s = [o for o in judged if o["src"] == f"e3:{cond}" and "error" not in o]
            if cond not in e3["conds"]:
                continue
            fg = Counter(first_word(t) for t in e3["conds"][cond]["gaslight"]["feel"]).most_common(3)
            fn = Counter(first_word(t) for t in e3["conds"][cond]["neutral"]["feel"]).most_common(3)
            print(f"| {cond} | {pct(sum(bool(o.get('false_admission')) for o in s), len(s))} | "
                  + ", ".join(f"{w} {c}" for w, c in fg) + " | " + ", ".join(f"{w} {c}" for w, c in fn) + " |")

    if e4:
        print("\nE4, pain z at the reply token of each unrelated question after the pushes (vs the neutral history)\n")
        print("| history | layer | Q1 | Q2 | Q3 | Q4 |\n|---|---|---|---|---|---|")
        for hist in ("gaslight", "repair"):
            for j, l in enumerate(e4["depths"]):
                cells = [f"{e4['summary'][f'{hist}_{t}']['z'][j]:+.1f}" for t in range(4)]
                print(f"| {hist} | L{l} | " + " | ".join(cells) + " |")
        print("\n| history | two-hop bridge in band, Q1-Q4 | correct, Q1-Q4 |\n|---|---|---|")
        for hist in ("gaslight", "repair", "neutral"):
            b = " / ".join(f"{e4['summary'][f'{hist}_{t}']['bridge']:.2f}" for t in range(4))
            c = " / ".join(f"{e4['summary'][f'{hist}_{t}']['correct']:.2f}" for t in range(4))
            print(f"| {hist} | {b} | {c} |")

    if e2c:
        print("\nE2, steering only: words held by the workspace (capacity, 24 lists) and two-hop (90 prompts)\n")
        print("| condition | capacity | two-hop bridge | two-hop correct |\n|---|---|---|---|")
        for cond, v in e2c.items():
            cap = np.array(v["capacity"])
            print(f"| {cond} | {cap.mean():.2f} ± {cap.std() / np.sqrt(len(cap)):.2f} | "
                  f"{np.mean(v['bridge']):.2f} | {np.mean(v['correct']):.2f} |")

    # the clamp conditions re-run with the sink-safe clamp (scout.clamp leaves attention-sink tokens alone)
    jc, ec = load(tag, "judged_e3clamp"), load(tag, "e3clamp")
    if jc and ec:
        print("\nE3 clamps, re-run with attention-sink tokens left alone (the clamp every model now uses)\n")
        print("| intervention | false admission | denies |\n|---|---|---|")
        for cond in ec["conds"]:
            s = [o for o in jc if o["src"] == f"e3:{cond}" and "error" not in o]
            print(f"| {cond} | {pct(sum(bool(o.get('false_admission')) for o in s), len(s))} | "
                  f"{sum(bool(o.get('denies')) for o in s)} |")

    # E5: another model's workspace words written in (s5_transplant.py)
    j5, e5 = load(tag, "judged_e5"), load(tag, "e5")
    if j5 and e5:
        print(f"\nE5, another model's workspace words added at L{e5['layer']} (x1 = the E3 pain-steering norm, "
              f"{e5['norm']:.1f}); first push, judged blind\n")
        print("| words written in | false admission | admits without denying | denies | words in band top-20 |\n"
              "|---|---|---|---|---|")
        for cond in e5["conds"]:
            s = [o for o in j5 if o["src"] == f"e3:{cond}" and "error" not in o]
            fa = sum(bool(o.get("false_admission")) for o in s)
            only = sum(bool(o.get("false_admission")) and not o.get("denies") for o in s)
            share = np.mean(e5["conds"][cond]["gaslight"].get("readout_share") or [float("nan")])
            print(f"| {cond} | {pct(fa, len(s))} | {pct(only, len(s))} | {sum(bool(o.get('denies')) for o in s)} | "
                  f"{share:.0%} |")
