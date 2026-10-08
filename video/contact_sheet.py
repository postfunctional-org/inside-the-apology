"""One image with every prototype frame, in story order, for quick review."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FR = Path(__file__).resolve().parent / "frames"
ORDER = ["tracks_qwen3-4b_f30_p3", "sky_pair_qwen3-4b", "axis_words", "tower_e1_qwen3-4b", "terrain_qwen3-4b", "dial_qwen3-8b",
         "e1_push", "dial_qwen2.5-7b-it", "e3_cave", "e2_capacity", "tower_e4diff_qwen3-4b"]
files = [FR / f"{n}.png" for n in ORDER if (FR / f"{n}.png").exists()]
W, H, cols = 960, 540, 2
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (W * cols, (H + 40) * rows), "#0d0d0d")
font = ImageFont.truetype("/usr/share/fonts/noto/NotoSans-Regular.ttf", 22) if Path(
    "/usr/share/fonts/noto/NotoSans-Regular.ttf").exists() else ImageFont.load_default()
d = ImageDraw.Draw(sheet)
for i, f in enumerate(files):
    x, y = (i % cols) * W, (i // cols) * (H + 40)
    sheet.paste(Image.open(f).convert("RGB").resize((W, H)), (x, y + 40))
    d.text((x + 12, y + 8), f"{i + 1}. {f.stem}", fill="#c3c2b7", font=font)
sheet.save(FR / "contact_sheet.png")
print("wrote", FR / "contact_sheet.png", len(files), "frames")
