import sys
from pathlib import Path

sys.path.insert(0, str(Path("backend").resolve()))

import app.generators.lexicon as lex
import app.generators.phonetics as phon

print("=== VERIFYING FIXES ===")

# 1. гыйлем
plural_gilem = phon.inflect_plural("гыйлем")
print(f"гыйлем plural: {plural_gilem}")
assert plural_gilem == "ГЫЙЛЕМНӘР", f"Expected ГЫЙЛЕМНӘР, got {plural_gilem}"

# 2. сәгать
sagat_p1 = phon.inflect_possessive("сәгать", 1)
sagat_p3 = phon.inflect_possessive("сәгать", 3)
sagat_p3_dat = phon.inflect_possessive_case("сәгать", 3, "case_dative")
print(f"сәгать: {sagat_p1}, {sagat_p3}, {sagat_p3_dat}")
assert sagat_p1 == "СӘГАТЕМ", f"Expected СӘГАТЕМ, got {sagat_p1}"
assert sagat_p3 == "СӘГАТЕ", f"Expected СӘГАТЕ, got {sagat_p3}"
assert sagat_p3_dat == "СӘГАТЕНӘ", f"Expected СӘГАТЕНӘ, got {sagat_p3_dat}"

# 3. кәгазь
kagaz_p1 = phon.inflect_possessive("кәгазь", 1)
kagaz_p3 = phon.inflect_possessive("кәгазь", 3)
kagaz_p3_acc = phon.inflect_possessive_case("кәгазь", 3, "case_accusative")
print(f"кәгазь: {kagaz_p1}, {kagaz_p3}, {kagaz_p3_acc}")
assert kagaz_p1 == "КӘГАЗЕМ", f"Expected КӘГАЗЕМ, got {kagaz_p1}"
assert kagaz_p3 == "КӘГАЗЕ", f"Expected КӘГАЗЕ, got {kagaz_p3}"
assert kagaz_p3_acc == "КӘГАЗЕН", f"Expected КӘГАЗЕН, got {kagaz_p3_acc}"

# 4. тау
tau_p1 = phon.inflect_possessive("тау", 1)
tau_p3 = phon.inflect_possessive("тау", 3)
tau_p3_abl = phon.inflect_possessive_case("тау", 3, "case_ablative")
print(f"тау: {tau_p1}, {tau_p3}, {tau_p3_abl}")
assert tau_p1 == "ТАВЫМ", f"Expected ТАВЫМ, got {tau_p1}"
assert tau_p3 == "ТАВЫ", f"Expected ТАВЫ, got {tau_p3}"
assert tau_p3_abl == "ТАВЫННАН", f"Expected ТАВЫННАН, got {tau_p3_abl}"

# 5. сорау
sorau_p1 = phon.inflect_possessive("сорау", 1)
sorau_p3 = phon.inflect_possessive("сорау", 3)
print(f"сорау: {sorau_p1}, {sorau_p3}")
assert sorau_p1 == "СОРАВЫМ", f"Expected СОРАВЫМ, got {sorau_p1}"
assert sorau_p3 == "СОРАВЫ", f"Expected СОРАВЫ, got {sorau_p3}"

# 6. су
su_p1 = phon.inflect_possessive("су", 1)
su_p3 = phon.inflect_possessive("су", 3)
su_p3_loc = phon.inflect_possessive_case("су", 3, "case_locative")
print(f"су: {su_p1}, {su_p3}, {su_p3_loc}")
assert su_p1 == "СУЫМ", f"Expected СУЫМ, got {su_p1}"
assert su_p3 == "СУЫ", f"Expected СУЫ, got {su_p3}"
assert su_p3_loc == "СУЫНДА", f"Expected СУЫНДА, got {su_p3_loc}"

# 7. аю
ayu_p1 = phon.inflect_possessive("аю", 1)
ayu_p3 = phon.inflect_possessive("аю", 3)
print(f"аю: {ayu_p1}, {ayu_p3}")
assert ayu_p1 == "АЮЫМ", f"Expected АЮЫМ, got {ayu_p1}"
assert ayu_p3 == "АЮЫ", f"Expected АЮЫ, got {ayu_p3}"

# 8. word order
assert lex.WORD_ORDER_SETS[3][1] == "БГВА", f"Expected БГВА, got {lex.WORD_ORDER_SETS[3][1]}"

# 9. compound words
for p1, p2, comp in lex.COMPOUND_WORDS:
    assert comp in ["КУЛЪЯУЛЫК", "АККОШ", "ӨЧПОЧМАК", "КӨНЧЫГЫШ", "КӨНБАТЫШ", "БАЛКОРТЫ", "ТАШКҮМЕР", "БИЛБАВЫ", "КЫЗЫЛТҮШ", "ТӨНБОЕК", "ГӨЛҖИМЕШ", "ИТТАРТКЫЧ"]

print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")
