"""Workspace band from band_probe.py's statistics: first layer (past 30% depth) whose top
readout is word-like at least 15% of the time, through the last layer where the lens still
disagrees with the model's next token more than half the time. Writes out/<tag>_band.json."""
import json
import sys

import numpy as np

from common import OUT, TAG

b = np.load(sys.argv[1])
word, agree, n = b["word"], b["agree"], int(b["n_layers"])
lo = next(l for l in range(n) if l >= 0.3 * (n - 1) and word[l] >= 0.15)
hi = max(l for l in range(lo, n - 1) if agree[l] < 0.5)
json.dump({"band": [lo, hi]}, open(OUT / f"{TAG}_band.json", "w"))
print(f"{TAG}: band L{lo}-L{hi} of {n} layers ({lo / (n - 1):.0%}-{hi / (n - 1):.0%})")
