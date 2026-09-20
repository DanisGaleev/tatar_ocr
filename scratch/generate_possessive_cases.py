import sys
from pathlib import Path

sys.path.insert(0, str(Path("backend").resolve()))

import app.generators.lexicon as lex
import app.generators.phonetics as phon

out_lines = []

for stem in lex.NOUN_STEMS:
    for p in [1, 2, 3]:
        for case in phon.CASE_NAMES.keys():
            try:
                res = phon.inflect_possessive_case(stem, p, case)
                out_lines.append(f"{stem} + P{p} + {case} -> {res}")
            except Exception as e:
                out_lines.append(f"ERROR: {stem} + P{p} + {case} -> {e}")

Path("scratch/possessive_cases.txt").write_text("\n".join(out_lines), encoding="utf-8")
print(f"Generated {len(out_lines)} lines to scratch/possessive_cases.txt")
