import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

TATAR_SPEC = "АӘБВГДЕЁЖҖЗИЙКЛМНҢОӨПРСТУҮФХҺЦЧШЩЪЫЬЭЮЯаәбвгдеёжҗзийклмнңоөпрстуүфхһцчшщъыьэюя"
FONTS_DIR = Path("fonts")

def test_font(font_path):
    try:
        font = ImageFont.truetype(str(font_path), 40)
    except Exception as e:
        return f"Error loading: {e}"
    
    missing = []
    for ch in TATAR_SPEC:
        img = Image.new("L", (60, 60), 255)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), ch, font=font, fill=0)
        arr = np.array(img)
        if (arr < 200).sum() == 0:
            missing.append(ch)
            continue
        bbox = draw.textbbox((0, 0), ch, font=font)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            missing.append(ch)
    return missing

print("=== Checking local fonts ===")
for f in sorted(FONTS_DIR.glob("*.[to]tf")):
    res = test_font(f)
    print(f"{f.name}: missing {len(res)} chars -> {res[:10]}")

print("\n=== Checking Windows fonts ===")
win_fonts = [
    "C:/Windows/Fonts/comic.ttf",
    "C:/Windows/Fonts/comicbd.ttf",
    "C:/Windows/Fonts/segoepr.ttf",
    "C:/Windows/Fonts/segoeprb.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/cambria.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "C:/Windows/Fonts/trebuc.ttf",
    "C:/Windows/Fonts/georgia.ttf",
    "C:/Windows/Fonts/consola.ttf",
]
for f in win_fonts:
    if os.path.exists(f):
        res = test_font(f)
        print(f"{Path(f).name}: missing {len(res)} chars -> {res[:10]}")
