import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

def inspect_tatar_chars(font_path, font_name):
    font = ImageFont.truetype(font_path, 32)
    chars = ["Ә", "Ө", "Ү", "Җ", "Ң", "Һ", "А", "О", "У", "Ж", "Н", "Х"]
    res = {}
    for ch in chars:
        im = Image.new("L", (48, 48), 255)
        d = ImageDraw.Draw(im)
        d.text((4, 4), ch, font=font, fill=0)
        arr = np.array(im)
        ink = (arr < 200).sum()
        bbox = d.textbbox((0, 0), ch, font=font)
        res[ch] = (ink, bbox)
    
    print(f"Font: {font_name}")
    for c in ["Ә", "Ө", "Ү", "Җ", "Ң", "Һ"]:
        print(f"  U+{ord(c):04X} ({c}): ink={res[c][0]}, bbox={res[c][1]}")

for fp in ["fonts/PlaypenSans.ttf", "fonts/Pangolin.ttf", "C:/Windows/Fonts/comic.ttf", "C:/Windows/Fonts/segoepr.ttf", "C:/Windows/Fonts/arial.ttf"]:
    if os.path.exists(fp):
        inspect_tatar_chars(fp, Path(fp).name)
