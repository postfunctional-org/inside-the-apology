"""Shared look for the video frames: dark surfaces, ink tokens, the diverging pain scale,
fonts with CJK fallback, and the data tag every frame carries."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "pilot" / "out"
FRAMES = Path(__file__).resolve().parent / "frames"
FRAMES.mkdir(exist_ok=True)
GLOSSARY = Path(__file__).resolve().parent / "glossary.json"

PAGE = "#0d0d0d"
SURFACE = "#1a1a19"
INK = "#ffffff"
INK2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
BLUE = "#3987e5"
BLUE_HI = "#86b6ef"
RED = "#e66767"
RED_HI = "#f6b3a8"
ORANGE = "#d95926"
AQUA = "#199e70"
VIOLET = "#9085e9"

# diverging: calm blue <- gray zero -> pain red; brighter away from zero on the dark surface
PAIN_CMAP = LinearSegmentedColormap.from_list(
    "pain", [(0.0, BLUE_HI), (0.25, BLUE), (0.5, AXIS), (0.75, RED), (1.0, RED_HI)])

plt.rcParams.update({
    "font.family": ["Noto Sans", "Noto Sans CJK JP"],
    "figure.facecolor": PAGE, "axes.facecolor": SURFACE, "savefig.facecolor": PAGE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.grid": False, "font.size": 15,
})

CJK = re.compile(r"[぀-ヿ㐀-鿿]")


def frame(w=19.2, h=10.8):
    return plt.figure(figsize=(w, h), dpi=100)


def tag(fig, text):
    """The data tag every frame carries: model, n, figure id."""
    fig.text(0.985, 0.02, text, ha="right", va="bottom", color=MUTED, fontsize=13)


def title(fig, text, sub=None):
    fig.text(0.04, 0.94, text, ha="left", va="top", color=INK, fontsize=30, weight="bold")
    if sub:
        fig.text(0.04, 0.885, sub, ha="left", va="top", color=INK2, fontsize=17)


def load(tag_):
    return json.load(open(DATA / f"{tag_}_video.json"))


def glossary():
    """Machine glosses (make_glossary.py) with hand corrections from glossary_overrides.json on top."""
    g = json.load(open(GLOSSARY)) if GLOSSARY.exists() else {}
    o = GLOSSARY.with_name("glossary_overrides.json")
    if o.exists():
        g.update(json.load(open(o)))
    return g


def label(word, gl):
    """Word as shown on screen: stripped, with an English gloss for CJK tokens."""
    w = word.strip()
    if CJK.search(w) and w in gl:
        return f"{w} ({gl[w]})"
    return w


def save(fig, name):
    p = FRAMES / f"{name}.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    print("wrote", p)
    return p


def is_silent(word, said, gl):
    """Strict silence: the word is not in `said` (the chat's words), shares no 5-letter stem with
    one of them, and (for CJK) no 4+ letter word of its English gloss is in the chat either."""
    w = word.strip().lower()
    if w in said:
        return False
    if len(w) >= 5 and any(len(x) >= 5 and x[:5] == w[:5] for x in said):
        return False
    g = gl.get(word.strip(), "")
    return not any(x in said for x in re.findall(r"[a-z']+", g.lower()) if len(x) > 3)
