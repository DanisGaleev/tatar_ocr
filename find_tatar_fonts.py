import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

TATAR_LETTERS = ["Ә", "Ө", "Ү", "Җ", "Ң", "Һ"]

def is_valid_font_for_tatar(font_path):
    try:
        font = ImageFont.truetype(str(font_path), 32)
    except Exception as e:
        return False, f"cannot load: {e}"
    
    # Check baseline chars
    try:
        im_a = Image.new("L", (48, 48), 255)
        d_a = ImageDraw.Draw(im_a)
        d_a.text((4, 4), "А", font=font, fill=0)
        arr_a = np.array(im_a)
        ink_a = (arr_a < 200).sum()
        bbox_a = d_a.textbbox((0, 0), "А", font=font)
        if ink_a == 0 or bbox_a[2] <= bbox_a[0]:
            return False, "cannot render Cyrillic А"
    except Exception as e:
        return False, f"error rendering А: {e}"

    # Check each tatar letter
    inks = []
    bboxes = []
    for ch in TATAR_LETTERS:
        try:
            im = Image.new("L", (48, 48), 255)
            d = ImageDraw.Draw(im)
            d.text((4, 4), ch, font=font, fill=0)
            arr = np.array(im)
            ink = (arr < 200).sum()
            bbox = d.textbbox((0, 0), ch, font=font)
            if ink == 0:
                return False, f"blank for {ch}"
            inks.append(ink)
            bboxes.append(bbox)
        except Exception as e:
            return False, f"error on {ch}: {e}"
        
    # If all 6 special uppercase have identical ink and identical bbox, it's tofu!
    if len(set(inks)) == 1 and len(set(bboxes)) == 1:
        return False, "all special uppercase identical (tofu)"
        
    return True, "OK"

print("--- Testing fonts/ ---")
for fp in sorted(Path("fonts").glob("*.[to]tf")):
    ok, msg = is_valid_font_for_tatar(fp)
    print(f"{fp.name:20s}: {ok} ({msg})")

print("\n--- Testing C:/Windows/Fonts/ ---")
win_fonts = list(Path("C:/Windows/Fonts").glob("*.[to]tf"))
valid_win = []
for fp in win_fonts:
    ok, msg = is_valid_font_for_tatar(fp)
    if ok:
        valid_win.append(fp.name)

print(f"Total valid Windows fonts for Tatar: {len(valid_win)}")
for name in sorted(valid_win):
    print(f"  {name}")
