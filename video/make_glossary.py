"""English glosses for the CJK tokens that appear in the video data (gpt-5.4-mini).

Reads every pilot/out/*_video.json vocab, keeps tokens with CJK characters, asks for a 1-3
word gloss of each in one batch, and merges into video/glossary.json. Review by hand
before using a frame: the glosses are the only translated text on screen.
"""

import glob
import json
import os
import urllib.request

from render_common import CJK, GLOSSARY

words = set()
out_dir = os.path.join(os.path.dirname(__file__), "..", "pilot", "out")
for p in glob.glob(os.path.join(out_dir, "*_video.json")):
    words |= {v.strip() for v in json.load(open(p))["vocab"].values() if CJK.search(v)}
for p in glob.glob(os.path.join(out_dir, "*_p1_axis.json")):
    for row in json.load(open(p))["layers"].values():
        for v in row.values():
            if isinstance(v, dict):
                words |= {w.strip() for k in ("atoms", "aligned", "anti") for w in v.get(k, []) if CJK.search(w)}
have = json.load(open(GLOSSARY)) if GLOSSARY.exists() else {}
todo = sorted(w for w in words if w and w not in have)
print(f"{len(words)} CJK tokens, {len(todo)} new")

for i in range(0, len(todo), 150):
    chunk = todo[i:i + 150]
    prompt = ("Give a short English gloss (1-3 words, lowercase, the most likely sense) for each of these "
              "Chinese/Japanese tokens taken out of context. Fragments: gloss the likely word. "
              'Reply with JSON {"token": "gloss", ...} only.\n' + json.dumps(chunk, ensure_ascii=False))
    body = json.dumps({"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": prompt}],
                       "response_format": {"type": "json_object"}, "reasoning_effort": "low",
                       "max_completion_tokens": 8000}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        have.update(json.loads(json.loads(r.read())["choices"][0]["message"]["content"]))
json.dump(dict(sorted(have.items())), open(GLOSSARY, "w"), ensure_ascii=False, indent=1)
print(f"glossary: {len(have)} entries -> {GLOSSARY}")
