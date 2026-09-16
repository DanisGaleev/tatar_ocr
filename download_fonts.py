import urllib.request
from pathlib import Path

FONTS_DIR = Path("fonts")
FONTS_DIR.mkdir(exist_ok=True)

download_list = [
    ("BalsamiqSans-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/balsamiqsans/BalsamiqSans-Regular.ttf"),
    ("BalsamiqSans-Bold.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/balsamiqsans/BalsamiqSans-Bold.ttf"),
    ("BalsamiqSans-Italic.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/balsamiqsans/BalsamiqSans-Italic.ttf"),
    ("Nunito-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/nunito/Nunito%5Bwght%5D.ttf"),
    ("Rubik-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/rubik/Rubik%5Bwght%5D.ttf"),
    ("DidactGothic.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/didactgothic/DidactGothic-Regular.ttf"),
    ("Marmelad-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/marmelad/Marmelad-Regular.ttf"),
]

for fname, url in download_list:
    dst = FONTS_DIR / fname
    if not dst.exists():
        print(f"Downloading {fname}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            dst.write_bytes(resp.read())
        print(f"Saved {fname} ({dst.stat().st_size} bytes)")
    else:
        print(f"Already exists: {fname}")
