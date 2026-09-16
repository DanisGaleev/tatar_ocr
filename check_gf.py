import urllib.request
import json
from pathlib import Path
from fontTools.ttLib import TTFont
import io

TATAR_SPEC = "АӘБВГДЕЁЖҖЗИЙКЛМНҢОӨПРСТУҮФХҺЦЧШЩЪЫЬЭЮЯаәбвгдеёжҗзийклмнңоөпрстуүфхһцчшщъыьэюя0123456789.,!?-:;\"'()«»—"
TATAR_POINTS = set(ord(c) for c in TATAR_SPEC)

# List candidate handwriting/casual/print fonts from Google Fonts
candidates = [
    ("BalsamiqSans", "https://raw.githubusercontent.com/google/fonts/main/ofl/balsamiqsans/BalsamiqSans-Regular.ttf"),
    ("BalsamiqSans-Bold", "https://raw.githubusercontent.com/google/fonts/main/ofl/balsamiqsans/BalsamiqSans-Bold.ttf"),
    ("Neucha", "https://raw.githubusercontent.com/google/fonts/main/ofl/neucha/Neucha-Regular.ttf"),
    ("ShantellSans", "https://raw.githubusercontent.com/google/fonts/main/ofl/shantellsans/ShantellSans%5BBNCE%2Cital%2CINFM%2Cwght%5D.ttf"),
    ("ComicNeue", "https://raw.githubusercontent.com/google/fonts/main/ofl/comicneue/ComicNeue-Regular.ttf"),
    ("MarckScript", "https://raw.githubusercontent.com/google/fonts/main/ofl/marckscript/MarckScript-Regular.ttf"),
    ("Underdog", "https://raw.githubusercontent.com/google/fonts/main/ofl/underdog/Underdog-Regular.ttf"),
    ("KellySlab", "https://raw.githubusercontent.com/google/fonts/main/ofl/kellyslab/KellySlab-Regular.ttf"),
    ("Rubik", "https://raw.githubusercontent.com/google/fonts/main/ofl/rubik/Rubik%5Bwght%5D.ttf"),
    ("Nunito", "https://raw.githubusercontent.com/google/fonts/main/ofl/nunito/Nunito%5Bwght%5D.ttf"),
    ("Marmelad", "https://raw.githubusercontent.com/google/fonts/main/ofl/marmelad/Marmelad-Regular.ttf"),
    ("AmaticSC", "https://raw.githubusercontent.com/google/fonts/main/ofl/amaticsc/AmaticSC-Regular.ttf"),
    ("Caveat-Bold", "https://raw.githubusercontent.com/google/fonts/main/ofl/caveat/Caveat%5Bwght%5D.ttf"),
    ("RuslanDisplay", "https://raw.githubusercontent.com/google/fonts/main/ofl/ruslandisplay/RuslanDisplay-Regular.ttf"),
    ("SeymourOne", "https://raw.githubusercontent.com/google/fonts/main/ofl/seymourone/SeymourOne-Regular.ttf"),
]

for name, url in candidates:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read()
        font = TTFont(io.BytesIO(data))
        chars = set()
        for t in font['cmap'].tables:
            if t.isUnicode():
                chars.update(t.cmap.keys())
        missing = [c for c in TATAR_SPEC if ord(c) not in chars]
        print(f"{name:20s}: size={len(data):6d}, missing={len(missing)} ({''.join(missing)})")
    except Exception as e:
        print(f"{name:20s}: error: {e}")
