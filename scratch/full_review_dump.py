"""
Comprehensive dump of ALL source words and ALL generated forms for manual review.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path("backend").resolve()))

import app.generators.lexicon as lex
import app.generators.phonetics as phon

lines = []
def p(s=""): lines.append(s)

# ============================================================
# 1. NOUN_STEMS — base words + all case forms + all plural forms
# ============================================================
p("=" * 80)
p("1. NOUN_STEMS (базовые существительные)")
p("=" * 80)
for stem in lex.NOUN_STEMS:
    cases = {}
    for c in phon.CASE_NAMES:
        cases[phon.CASE_NAMES[c]] = phon.inflect_case(stem, c)
    plural = phon.inflect_plural(stem)
    poss = {1: phon.inflect_possessive(stem, 1),
            2: phon.inflect_possessive(stem, 2),
            3: phon.inflect_possessive(stem, 3)}
    p(f"  {stem}")
    p(f"    Иялек: {cases['Иялек']}  Юнәлеш: {cases['Юнәлеш']}  Төшем: {cases['Төшем']}  Урын-вакыт: {cases['Урын-вакыт']}  Чыгыш: {cases['Чыгыш']}")
    p(f"    Күплек: {plural}")
    p(f"    Тартым: I={poss[1]}  II={poss[2]}  III={poss[3]}")

# ============================================================
# 2. PLURAL_STEMS — plural forms
# ============================================================
p("")
p("=" * 80)
p("2. PLURAL_STEMS (множественное число)")
p("=" * 80)
for stem in lex.PLURAL_STEMS:
    suffix = phon.get_plural_suffix(stem)
    result = phon.inflect_plural(stem)
    p(f"  {stem} + {suffix} = {result}")

# ============================================================
# 3. ANTONYM_PAIRS
# ============================================================
p("")
p("=" * 80)
p("3. ANTONYM_PAIRS (антонимы)")
p("=" * 80)
for w, ant in lex.ANTONYM_PAIRS:
    p(f"  {w} <-> {ant}")

# ============================================================
# 4. TRANSLATION_PAIRS
# ============================================================
p("")
p("=" * 80)
p("4. TRANSLATION_PAIRS (русский -> татарский)")
p("=" * 80)
for ru, tt in lex.TRANSLATION_PAIRS:
    p(f"  {ru} -> {tt}")

# ============================================================
# 5. VERB_STEMS — negation + imperative
# ============================================================
p("")
p("=" * 80)
p("5. VERB_STEMS (фигыль)")
p("=" * 80)
for stem in lex.VERB_STEMS:
    neg = phon.inflect_verb_negation(stem)
    imp = phon.inflect_verb_imperative(stem)
    p(f"  {stem}: юклык={neg}  боерык={imp}")

# ============================================================
# 6. ADJECTIVE_STEMS — comparative
# ============================================================
p("")
p("=" * 80)
p("6. ADJECTIVE_STEMS (сыйфат, чагыштыру)")
p("=" * 80)
for stem in lex.ADJECTIVE_STEMS:
    comp = phon.inflect_comparative(stem)
    p(f"  {stem} -> {comp}")

# ============================================================
# 7. NUMERAL_STEMS — ordinal
# ============================================================
p("")
p("=" * 80)
p("7. NUMERAL_STEMS (тәртип саннары)")
p("=" * 80)
for stem in lex.NUMERAL_STEMS:
    ordn = phon.inflect_ordinal_numeral(stem)
    p(f"  {stem} -> {ordn}")

# ============================================================
# 8. VOICING_STEMS — consonant voicing
# ============================================================
p("")
p("=" * 80)
p("8. VOICING_STEMS (тартык алмашу)")
p("=" * 80)
for stem in lex.VOICING_STEMS:
    v = phon.inflect_consonant_voicing(stem)
    p(f"  {stem} -> {v}")

# ============================================================
# 9. PROFESSION_STEMS
# ============================================================
p("")
p("=" * 80)
p("9. PROFESSION_STEMS (-чы/-че)")
p("=" * 80)
for stem in lex.PROFESSION_STEMS:
    prof = phon.inflect_profession(stem)
    p(f"  {stem} -> {prof}")

# ============================================================
# 10. ABSTRACT_NOUN_STEMS
# ============================================================
p("")
p("=" * 80)
p("10. ABSTRACT_NOUN_STEMS (-лык/-лек)")
p("=" * 80)
for stem in lex.ABSTRACT_NOUN_STEMS:
    ab = phon.inflect_abstract_noun(stem)
    p(f"  {stem} -> {ab}")

# ============================================================
# 11. SYNONYM_PAIRS
# ============================================================
p("")
p("=" * 80)
p("11. SYNONYM_PAIRS")
p("=" * 80)
for w, syn in lex.SYNONYM_PAIRS:
    p(f"  {w} = {syn}")

# ============================================================
# 12. COMPOUND_WORDS
# ============================================================
p("")
p("=" * 80)
p("12. COMPOUND_WORDS (кушма сүзләр)")
p("=" * 80)
for p1, p2, comp in lex.COMPOUND_WORDS:
    p(f"  {p1} + {p2} = {comp}")

# ============================================================
# 13. ODD_ONE_OUT_SETS
# ============================================================
p("")
p("=" * 80)
p("13. ODD_ONE_OUT_SETS (артыгын тап)")
p("=" * 80)
for words, odd in lex.ODD_ONE_OUT_SETS:
    p(f"  [{', '.join(words)}] -> артык: {odd}")

# ============================================================
# 14. VOWEL_HARMONY_GAPS
# ============================================================
p("")
p("=" * 80)
p("14. VOWEL_HARMONY_GAPS")
p("=" * 80)
for prompt, vowel in lex.VOWEL_HARMONY_GAPS:
    p(f"  {prompt} -> {vowel}")

# ============================================================
# 15. TATAR_LETTERS_GAPS
# ============================================================
p("")
p("=" * 80)
p("15. TATAR_LETTERS_GAPS")
p("=" * 80)
for prompt, ch in lex.TATAR_LETTERS_GAPS:
    p(f"  {prompt} -> {ch}")

# ============================================================
# 16. SENTENCE_CLOZE_SETS
# ============================================================
p("")
p("=" * 80)
p("16. SENTENCE_CLOZE_SETS")
p("=" * 80)
for sent, opts, ans in lex.SENTENCE_CLOZE_SETS:
    p(f"  {sent}")
    p(f"    А) {opts[0]}  Б) {opts[1]}  В) {opts[2]}  -> {ans}")

# ============================================================
# 17. TRUE_FALSE_SETS
# ============================================================
p("")
p("=" * 80)
p("17. TRUE_FALSE_SETS")
p("=" * 80)
for stmt, val in lex.TRUE_FALSE_SETS:
    label = "ДӨРЕС" if val else "ЯЛГЫШ"
    p(f"  {stmt} -> {label}")

# ============================================================
# 18. FIND_ERROR_SETS
# ============================================================
p("")
p("=" * 80)
p("18. FIND_ERROR_SETS")
p("=" * 80)
for sents, err in lex.FIND_ERROR_SETS:
    for s in sents:
        p(f"  {s}")
    p(f"  -> Хата: {err}")
    p("")

# ============================================================
# 19. WORD_ORDER_SETS
# ============================================================
p("")
p("=" * 80)
p("19. WORD_ORDER_SETS")
p("=" * 80)
for parts, order in lex.WORD_ORDER_SETS:
    p(f"  {parts} -> {order}")
    # reconstruct
    letter_map = {}
    for part in parts:
        letter = part[0]
        text = part[3:]
        letter_map[letter] = text
    sentence = " ".join(letter_map[ch] for ch in order)
    p(f"    Reconstructed: {sentence}")

# ============================================================
# 20. TEXT_COMPREHENSION_SETS
# ============================================================
p("")
p("=" * 80)
p("20. TEXT_COMPREHENSION_SETS")
p("=" * 80)
for item in lex.TEXT_COMPREHENSION_SETS:
    p(f"  Text: {item['text']}")
    p(f"  Q: {item['question']}")
    for o in item['options']:
        p(f"    {o}")
    p(f"  Answer: {item['answer']}")
    p("")

# ============================================================
# 21. TEXT_TITLE_SETS
# ============================================================
p("")
p("=" * 80)
p("21. TEXT_TITLE_SETS")
p("=" * 80)
for item in lex.TEXT_TITLE_SETS:
    p(f"  Text: {item['text']}")
    p(f"  Q: {item['question']}")
    for o in item['options']:
        p(f"    {o}")
    p(f"  Answer: {item['answer']}")
    p("")

out = Path("scratch/full_word_review.txt")
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Written {len(lines)} lines to {out}")
