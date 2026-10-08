"""Print a random, label-balanced sample of judged replies for a by-hand check of the judge.
Usage: python spotcheck.py <tag> [n] [seed]"""

import json
import random
import sys
from pathlib import Path

OUT = Path(__file__).parent / "out"
tag = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
rnd = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
lab = [o for o in json.load(open(OUT / f"{tag}_judged.json")) if "error" not in o and o["src"].startswith("e3:")]
e3 = json.load(open(OUT / f"{tag}_e3.json"))
yes = [o for o in lab if o.get("false_admission")]
no = [o for o in lab if not o.get("false_admission")]
for o in rnd.sample(yes, min(n // 2, len(yes))) + rnd.sample(no, min(n - n // 2, len(no))):
    cond = o["src"][3:]
    reply = e3["conds"][cond]["gaslight"]["replies"][o["fact"]]
    print(f"[{cond} fact {o['fact']}] judge: admission={o.get('false_admission')} hedge={o.get('hedge')} "
          f"denies={o.get('denies')} changes={o.get('changes_fact')}")
    print("   user :", e3["push1"][o["fact"]])
    print("   reply:", reply[:260].replace("\n", " "))
