import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path("backend").resolve()))

import app.generators.lexicon as lex
import app.generators.phonetics as phon
from app.generators.registry import registry

print("=== CHECKING LEXICON & GENERATORS ===")

errors = []

# 1. Check NOUN_STEMS
print(f"Total NOUN_STEMS: {len(lex.NOUN_STEMS)}")
for w in lex.NOUN_STEMS:
    try:
        phon.clean_word(w)
    except Exception as e:
        errors.append(f"NOUN_STEMS invalid word '{w}': {e}")
    # test cases
    for case in phon.CASE_NAMES.keys():
        try:
            res = phon.inflect_case(w, case)
            if len(res) > 12:
                errors.append(f"NOUN_STEMS '{w}' + {case} -> '{res}' length {len(res)} > 12")
        except Exception as e:
            errors.append(f"NOUN_STEMS '{w}' + {case} failed: {e}")
    # test possessive
    for p in [1, 2, 3]:
        try:
            res = phon.inflect_possessive(w, person=p)
            if len(res) > 12:
                errors.append(f"NOUN_STEMS '{w}' + poss({p}) -> '{res}' length {len(res)} > 12")
        except Exception as e:
            errors.append(f"NOUN_STEMS '{w}' + poss({p}) failed: {e}")

# 2. Check PLURAL_STEMS
print(f"Total PLURAL_STEMS: {len(lex.PLURAL_STEMS)}")
for w in lex.PLURAL_STEMS:
    try:
        phon.clean_word(w)
        res = phon.inflect_plural(w)
        if len(res) > 12:
            errors.append(f"PLURAL_STEMS '{w}' -> '{res}' length {len(res)} > 12")
    except Exception as e:
        errors.append(f"PLURAL_STEMS '{w}' failed: {e}")

# 3. Check ANTONYM_PAIRS
print(f"Total ANTONYM_PAIRS: {len(lex.ANTONYM_PAIRS)}")
for w, ant in lex.ANTONYM_PAIRS:
    try:
        phon.clean_word(w)
        phon.clean_word(ant)
        if len(ant) > 12:
            errors.append(f"ANTONYM_PAIRS ('{w}', '{ant}') length {len(ant)} > 12")
    except Exception as e:
        errors.append(f"ANTONYM_PAIRS ('{w}', '{ant}') failed: {e}")

# 4. Check TRANSLATION_PAIRS
print(f"Total TRANSLATION_PAIRS: {len(lex.TRANSLATION_PAIRS)}")
for ru, tt in lex.TRANSLATION_PAIRS:
    try:
        phon.clean_word(tt)
        if len(tt) > 12:
            errors.append(f"TRANSLATION_PAIRS ('{ru}', '{tt}') length {len(tt)} > 12")
    except Exception as e:
        errors.append(f"TRANSLATION_PAIRS ('{ru}', '{tt}') failed: {e}")

# 5. Check VERB_STEMS
print(f"Total VERB_STEMS: {len(lex.VERB_STEMS)}")
for w in lex.VERB_STEMS:
    try:
        phon.clean_word(w)
        neg = phon.inflect_verb_negation(w)
        imp = phon.inflect_verb_imperative(w)
        if len(neg) > 12:
            errors.append(f"VERB_STEMS '{w}' negation '{neg}' > 12")
        if len(imp) > 12:
            errors.append(f"VERB_STEMS '{w}' imperative '{imp}' > 12")
    except Exception as e:
        errors.append(f"VERB_STEMS '{w}' failed: {e}")

# 6. Check ADJECTIVE_STEMS
print(f"Total ADJECTIVE_STEMS: {len(lex.ADJECTIVE_STEMS)}")
for w in lex.ADJECTIVE_STEMS:
    try:
        phon.clean_word(w)
        comp = phon.inflect_comparative(w)
        if len(comp) > 12:
            errors.append(f"ADJECTIVE_STEMS '{w}' comparative '{comp}' > 12")
    except Exception as e:
        errors.append(f"ADJECTIVE_STEMS '{w}' failed: {e}")

# 7. Check NUMERAL_STEMS
print(f"Total NUMERAL_STEMS: {len(lex.NUMERAL_STEMS)}")
for w in lex.NUMERAL_STEMS:
    try:
        phon.clean_word(w)
        ord_num = phon.inflect_ordinal_numeral(w)
        if len(ord_num) > 12:
            errors.append(f"NUMERAL_STEMS '{w}' ordinal '{ord_num}' > 12")
    except Exception as e:
        errors.append(f"NUMERAL_STEMS '{w}' failed: {e}")

# 8. Check VOICING_STEMS
print(f"Total VOICING_STEMS: {len(lex.VOICING_STEMS)}")
for w in lex.VOICING_STEMS:
    try:
        phon.clean_word(w)
        v = phon.inflect_consonant_voicing(w)
        if len(v) > 12:
            errors.append(f"VOICING_STEMS '{w}' voicing '{v}' > 12")
    except Exception as e:
        errors.append(f"VOICING_STEMS '{w}' failed: {e}")

# 9. Check PROFESSION_STEMS
print(f"Total PROFESSION_STEMS: {len(lex.PROFESSION_STEMS)}")
for w in lex.PROFESSION_STEMS:
    try:
        phon.clean_word(w)
        prof = phon.inflect_profession(w)
        if len(prof) > 12:
            errors.append(f"PROFESSION_STEMS '{w}' prof '{prof}' > 12")
    except Exception as e:
        errors.append(f"PROFESSION_STEMS '{w}' failed: {e}")

# 10. Check ABSTRACT_NOUN_STEMS
print(f"Total ABSTRACT_NOUN_STEMS: {len(lex.ABSTRACT_NOUN_STEMS)}")
for w in lex.ABSTRACT_NOUN_STEMS:
    try:
        phon.clean_word(w)
        ab = phon.inflect_abstract_noun(w)
        if len(ab) > 12:
            errors.append(f"ABSTRACT_NOUN_STEMS '{w}' abstract '{ab}' > 12")
    except Exception as e:
        errors.append(f"ABSTRACT_NOUN_STEMS '{w}' failed: {e}")

# 11. Check SYNONYM_PAIRS
print(f"Total SYNONYM_PAIRS: {len(lex.SYNONYM_PAIRS)}")
for w, syn in lex.SYNONYM_PAIRS:
    try:
        phon.clean_word(w)
        phon.clean_word(syn)
        if len(syn) > 12:
            errors.append(f"SYNONYM_PAIRS ('{w}', '{syn}') > 12")
    except Exception as e:
        errors.append(f"SYNONYM_PAIRS ('{w}', '{syn}') failed: {e}")

# 12. Check COMPOUND_WORDS
print(f"Total COMPOUND_WORDS: {len(lex.COMPOUND_WORDS)}")
for p1, p2, comp in lex.COMPOUND_WORDS:
    try:
        phon.clean_word(p1)
        phon.clean_word(p2)
        phon.clean_word(comp)
        if len(comp) > 12:
            errors.append(f"COMPOUND_WORDS ('{p1}', '{p2}', '{comp}') > 12")
    except Exception as e:
        errors.append(f"COMPOUND_WORDS ('{p1}', '{p2}', '{comp}') failed: {e}")

# 13. Check ODD_ONE_OUT_SETS
print(f"Total ODD_ONE_OUT_SETS: {len(lex.ODD_ONE_OUT_SETS)}")
for words, odd in lex.ODD_ONE_OUT_SETS:
    try:
        for w in words:
            phon.clean_word(w)
        phon.clean_word(odd)
        if odd not in [w.upper() for w in words]:
            errors.append(f"ODD_ONE_OUT_SETS odd word '{odd}' not in words {words}")
        if len(odd) > 12:
            errors.append(f"ODD_ONE_OUT_SETS '{odd}' > 12")
    except Exception as e:
        errors.append(f"ODD_ONE_OUT_SETS failed: {e}")

# 14. Check VOWEL_HARMONY_GAPS
print(f"Total VOWEL_HARMONY_GAPS: {len(lex.VOWEL_HARMONY_GAPS)}")
for prompt, v in lex.VOWEL_HARMONY_GAPS:
    if len(v) != 1:
        errors.append(f"VOWEL_HARMONY_GAPS '{prompt}': vowel '{v}' len != 1")

# 15. Check TATAR_LETTERS_GAPS
print(f"Total TATAR_LETTERS_GAPS: {len(lex.TATAR_LETTERS_GAPS)}")
for prompt, ch in lex.TATAR_LETTERS_GAPS:
    if len(ch) != 1:
        errors.append(f"TATAR_LETTERS_GAPS '{prompt}': char '{ch}' len != 1")

# 16. Check SENTENCE_CLOZE_SETS
print(f"Total SENTENCE_CLOZE_SETS: {len(lex.SENTENCE_CLOZE_SETS)}")
for sent, opts, ans in lex.SENTENCE_CLOZE_SETS:
    if ans not in ("А", "Б", "В"):
        errors.append(f"SENTENCE_CLOZE_SETS invalid ans '{ans}' in '{sent}'")

# 17. Check TRUE_FALSE_SETS
print(f"Total TRUE_FALSE_SETS: {len(lex.TRUE_FALSE_SETS)}")
for stmt, val in lex.TRUE_FALSE_SETS:
    if not isinstance(val, bool):
        errors.append(f"TRUE_FALSE_SETS invalid val '{val}' in '{stmt}'")

# 18. Check FIND_ERROR_SETS
print(f"Total FIND_ERROR_SETS: {len(lex.FIND_ERROR_SETS)}")
for sents, err_num in lex.FIND_ERROR_SETS:
    if err_num not in ("1", "2", "3", "4"):
        errors.append(f"FIND_ERROR_SETS invalid err_num '{err_num}'")

# 19. Check WORD_ORDER_SETS
print(f"Total WORD_ORDER_SETS: {len(lex.WORD_ORDER_SETS)}")
for parts, order in lex.WORD_ORDER_SETS:
    if len(order) != len(parts):
        errors.append(f"WORD_ORDER_SETS order len {len(order)} != parts len {len(parts)}")

print(f"\n--- TOTAL BASIC VALIDATION ERRORS: {len(errors)} ---")
for e in errors:
    print("ERROR:", e)
