import os
import sys
from pathlib import Path
from fontTools.ttLib import TTFont

sys.stdout.reconfigure(encoding='utf-8')

TATAR_SPEC = "АӘБВГДЕЁЖҖЗИЙКЛМНҢОӨПРСТУҮФХҺЦЧШЩЪЫЬЭЮЯаәбвгдеёжҗзийклмнңоөпрстуүфхһцчшщъыьэюя0123456789.,!?-:;\"'()«»—"

def get_font_characters(font_path):
    try:
        font = TTFont(font_path)
        characters = set()
        for table in font['cmap'].tables:
            if table.isUnicode():
                characters.update(table.cmap.keys())
        return characters
    except Exception as e:
        return set()

print("=== Checking fonts in fonts/ ===")
for p in sorted(Path("fonts").glob("*.[to]tf")):
    supported = get_font_characters(p)
    missing = [c for c in TATAR_SPEC if ord(c) not in supported]
    print(f"{p.name:20s}: missing {len(missing)} -> {''.join(missing)}")

print("\n=== Checking Windows fonts for full support ===")
supported_win = []
for p in sorted(Path("C:/Windows/Fonts").glob("*.[to]tf")):
    supported = get_font_characters(p)
    missing = [c for c in TATAR_SPEC if ord(c) not in supported]
    if len(missing) == 0:
        supported_win.append(p.name)

print(f"Windows fonts with 100% support for all {len(TATAR_SPEC)} Tatar chars: {len(supported_win)}")
for name in supported_win:
    print(f"  {name}")
