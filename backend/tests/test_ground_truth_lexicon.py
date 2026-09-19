import pytest
from app.generators.phonetics import inflect_case, inflect_plural, CASE_NAMES
from app.generators.case_inflection import DEFAULT_NOUN_STEMS
from app.generators.plural_affixes import DEFAULT_PLURAL_STEMS
from app.generators.antonyms import DEFAULT_ANTONYM_PAIRS
from app.generators.translation import DEFAULT_TRANSLATION_PAIRS

def test_verify_all_noun_stems():
    expected_matrix = {
        # Front vowels
        "дәфтәр": {"case_dative": "ДӘФТӘРГӘ", "case_ablative": "ДӘФТӘРДӘН", "case_locative": "ДӘФТӘРДӘ", "case_accusative": "ДӘФТӘРНЕ", "case_genitive": "ДӘФТӘРНЕҢ"},
        "өстәл": {"case_dative": "ӨСТӘЛГӘ", "case_ablative": "ӨСТӘЛДӘН", "case_locative": "ӨСТӘЛДӘ", "case_accusative": "ӨСТӘЛНЕ", "case_genitive": "ӨСТӘЛНЕҢ"},
        "мәктәп": {"case_dative": "МӘКТӘПКӘ", "case_ablative": "МӘКТӘПТӘН", "case_locative": "МӘКТӘПТӘ", "case_accusative": "МӘКТӘПНЕ", "case_genitive": "МӘКТӘПНЕҢ"},
        "күз": {"case_dative": "КҮЗГӘ", "case_ablative": "КҮЗДӘН", "case_locative": "КҮЗДӘ", "case_accusative": "КҮЗНЕ", "case_genitive": "КҮЗНЕҢ"},
        "иләк": {"case_dative": "ИЛӘККӘ", "case_ablative": "ИЛӘКТӘН", "case_locative": "ИЛӘКТӘ", "case_accusative": "ИЛӘКНЕ", "case_genitive": "ИЛӘКНЕҢ"},
        "өй": {"case_dative": "ӨЙГӘ", "case_ablative": "ӨЙДӘН", "case_locative": "ӨЙДӘ", "case_accusative": "ӨЙНЕ", "case_genitive": "ӨЙНЕҢ"},
        "җәй": {"case_dative": "ҖӘЙГӘ", "case_ablative": "ҖӘЙДӘН", "case_locative": "ҖӘЙДӘ", "case_accusative": "ҖӘЙНЕ", "case_genitive": "ҖӘЙНЕҢ"},
        "шәһәр": {"case_dative": "ШӘҺӘРГӘ", "case_ablative": "ШӘҺӘРДӘН", "case_locative": "ШӘҺӘРДӘ", "case_accusative": "ШӘҺӘРНЕ", "case_genitive": "ШӘҺӘРНЕҢ"},
        "кибет": {"case_dative": "КИБЕТКӘ", "case_ablative": "КИБЕТТӘН", "case_locative": "КИБЕТТӘ", "case_accusative": "КИБЕТНЕ", "case_genitive": "КИБЕТНЕҢ"},
        "күл": {"case_dative": "КҮЛГӘ", "case_ablative": "КҮЛДӘН", "case_locative": "КҮЛДӘ", "case_accusative": "КҮЛНЕ", "case_genitive": "КҮЛНЕҢ"},
        
        # Back vowels
        "китап": {"case_dative": "КИТАПКА", "case_ablative": "КИТАПТАН", "case_locative": "КИТАПТА", "case_accusative": "КИТАПНЫ", "case_genitive": "КИТАПНЫҢ"},
        "урман": {"case_dative": "УРМАНГА", "case_ablative": "УРМАННАН", "case_locative": "УРМАНДА", "case_accusative": "УРМАННЫ", "case_genitive": "УРМАННЫҢ"},
        "бала": {"case_dative": "БАЛАГА", "case_ablative": "БАЛАДАН", "case_locative": "БАЛАДА", "case_accusative": "БАЛАНЫ", "case_genitive": "БАЛАНЫҢ"},
        "алма": {"case_dative": "АЛМАГА", "case_ablative": "АЛМАДАН", "case_locative": "АЛМАДА", "case_accusative": "АЛМАНЫ", "case_genitive": "АЛМАНЫҢ"},
        "авыл": {"case_dative": "АВЫЛГА", "case_ablative": "АВЫЛДАН", "case_locative": "АВЫЛДА", "case_accusative": "АВЫЛНЫ", "case_genitive": "АВЫЛНЫҢ"},
        "сыйныф": {"case_dative": "СЫЙНЫФКА", "case_ablative": "СЫЙНЫФТАН", "case_locative": "СЫЙНЫФТА", "case_accusative": "СЫЙНЫФНЫ", "case_genitive": "СЫЙНЫФНЫҢ"},
        "болыт": {"case_dative": "БОЛЫТКА", "case_ablative": "БОЛЫТТАН", "case_locative": "БОЛЫТТА", "case_accusative": "БОЛЫТНЫ", "case_genitive": "БОЛЫТНЫҢ"},
        "таш": {"case_dative": "ТАШКА", "case_ablative": "ТАШТАН", "case_locative": "ТАШТА", "case_accusative": "ТАШНЫ", "case_genitive": "ТАШНЫҢ"},
        "таң": {"case_dative": "ТАҢГА", "case_ablative": "ТАҢНАН", "case_locative": "ТАҢДА", "case_accusative": "ТАҢНЫ", "case_genitive": "ТАҢНЫҢ"},
        "кул": {"case_dative": "КУЛГА", "case_ablative": "КУЛДАН", "case_locative": "КУЛДА", "case_accusative": "КУЛНЫ", "case_genitive": "КУЛНЫҢ"},
    }

    for stem, cases in expected_matrix.items():
        assert stem in DEFAULT_NOUN_STEMS
        for case_code, expected_word in cases.items():
            actual = inflect_case(stem, case_code)
            assert actual == expected_word, f"Mismatch for '{stem}' in '{case_code}': got '{actual}', expected '{expected_word}'"

def test_verify_all_plural_stems():
    expected_plurals = {
        # Standard harmony -лар / -ләр
        "бала": "БАЛАЛАР",
        "дәфтәр": "ДӘФТӘРЛӘР",
        "өй": "ӨЙЛӘР",
        "китап": "КИТАПЛАР",
        "тәрәзә": "ТӘРӘЗӘЛӘР",
        "мәктәп": "МӘКТӘПЛӘР",
        "күл": "КҮЛЛӘР",
        "шәһәр": "ШӘҺӘРЛӘР",
        "болыт": "БОЛЫТЛАР",
        "кибет": "КИБЕТЛӘР",
        "җырчы": "ҖЫРЧЫЛАР",
        # Nasal assimilation -нар / -нәр (м, н, ң)
        "урман": "УРМАННАР",
        "төн": "ТӨННӘР",
        "таң": "ТАҢНАР",
        "урам": "УРАМНАР",
        "көн": "КӨННӘР",
        "кием": "КИЕМНӘР",
        "каен": "КАЕННАР",
        "адым": "АДЫМНАР",
    }
    for stem, expected in expected_plurals.items():
        assert stem in DEFAULT_PLURAL_STEMS
        actual = inflect_plural(stem)
        assert actual == expected, f"Mismatch for plural '{stem}': got '{actual}', expected '{expected}'"

def test_verify_all_antonyms():
    for word, ant in DEFAULT_ANTONYM_PAIRS:
        assert len(ant) <= 12
        assert word
        assert ant.isupper()

def test_verify_all_translations():
    for ru, tt in DEFAULT_TRANSLATION_PAIRS:
        assert len(tt) <= 12
        assert ru
        assert tt.isupper()
