import urllib.request
import json
import sys
from fontTools.ttLib import TTFont
import io

sys.stdout.reconfigure(encoding='utf-8')
TATAR_SPEC = "АӘБВГДЕЁЖҖЗИЙКЛМНҢОӨПРСТУҮФХҺЦЧШЩЪЫЬЭЮЯаәбвгдеёжҗзийклмнңоөпрстуүфхһцчшщъыьэюя0123456789.,!?-:;\"'()«»—"

# Let's query GitHub API for google/fonts/tree/main/ofl
# Or check specific font families known to have handwritten / print / casual look
font_urls = {
    "BalsamiqSans-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/balsamiqsans/BalsamiqSans-Regular.ttf",
    "BalsamiqSans-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/balsamiqsans/BalsamiqSans-Bold.ttf",
    "BalsamiqSans-Italic": "https://raw.githubusercontent.com/google/fonts/main/ofl/balsamiqsans/BalsamiqSans-Italic.ttf",
    "Rubik-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/rubik/Rubik%5Bwght%5D.ttf",
    "Nunito-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/nunito/Nunito%5Bwght%5D.ttf",
    "Marmelad-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/marmelad/Marmelad-Regular.ttf",
    "Cuprum-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/cuprum/Cuprum%5Bwght%5D.ttf",
    "DidactGothic": "https://raw.githubusercontent.com/google/fonts/main/ofl/didactgothic/DidactGothic-Regular.ttf",
    "RussoOne": "https://raw.githubusercontent.com/google/fonts/main/ofl/russoone/RussoOne-Regular.ttf",
    "Exo2": "https://raw.githubusercontent.com/google/fonts/main/ofl/exo2/Exo2%5Bwght%5D.ttf",
    "Comfortaa": "https://raw.githubusercontent.com/google/fonts/main/ofl/comfortaa/Comfortaa%5Bwght%5D.ttf",
    "FiraSans": "https://raw.githubusercontent.com/google/fonts/main/ofl/firasans/FiraSans-Regular.ttf",
    "Ubuntu": "https://raw.githubusercontent.com/google/fonts/main/ofl/ubuntu/Ubuntu-Regular.ttf",
}

for name, url in font_urls.items():
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
        font = TTFont(io.BytesIO(data))
        chars = set()
        for t in font['cmap'].tables:
            if t.isUnicode():
                chars.update(t.cmap.keys())
        missing = [c for c in TATAR_SPEC if ord(c) not in chars]
        print(f"{name:20s}: missing={len(missing)}")
    except Exception as e:
        print(f"{name:20s}: error: {e}")
