import sys
from pathlib import Path

sys.path.insert(0, str(Path("backend").resolve()))

import app.generators.lexicon as lex
import app.generators.phonetics as phon

out_lines = []

def p(s=""):
    out_lines.append(s)

p("=== 1. NOUN INFLECTIONS (CASE & POSSESSIVE) ===")
for stem in lex.NOUN_STEMS:
    cases = [phon.inflect_case(stem, c) for c in phon.CASE_NAMES.keys()]
    poss = [phon.inflect_possessive(stem, p) for p in [1, 2, 3]]
    p(f"{stem} | Cases: {', '.join(cases)} | Poss: {', '.join(poss)}")

p("\n=== 2. PLURAL FORMS ===")
for stem in lex.PLURAL_STEMS:
    p(f"{stem} -> {phon.inflect_plural(stem)}")

p("\n=== 3. VERBS ===")
for stem in lex.VERB_STEMS:
    neg = phon.inflect_verb_negation(stem)
    imp = phon.inflect_verb_imperative(stem)
    p(f"{stem} | Neg: {neg} | Imp: {imp}")

p("\n=== 4. ADJECTIVES (COMPARATIVE) ===")
for stem in lex.ADJECTIVE_STEMS:
    p(f"{stem} -> {phon.inflect_comparative(stem)}")

p("\n=== 5. NUMERALS (ORDINAL) ===")
for stem in lex.NUMERAL_STEMS:
    p(f"{stem} -> {phon.inflect_ordinal_numeral(stem)}")

p("\n=== 6. PROFESSIONS & ABSTRACT NOUNS ===")
for stem in lex.PROFESSION_STEMS:
    p(f"Prof: {stem} -> {phon.inflect_profession(stem)}")
for stem in lex.ABSTRACT_NOUN_STEMS:
    p(f"Abstract: {stem} -> {phon.inflect_abstract_noun(stem)}")

p("\n=== 7. CONSONANT VOICING ===")
for stem in lex.VOICING_STEMS:
    p(f"Voicing: {stem} -> {phon.inflect_consonant_voicing(stem)}")

output_path = Path("scratch/all_inflected_words.txt")
output_path.write_text("\n".join(out_lines), encoding="utf-8")
print(f"Written {len(out_lines)} lines to {output_path}")
