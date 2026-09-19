import pytest
from app.generators.phonetics import (
    clean_word,
    get_last_vowel,
    is_front_vowel,
    get_case_suffix,
    inflect_case,
    get_plural_suffix,
    inflect_plural,
    build_expected_cells,
    PhoneticsError,
)

class TestTatarPhoneticsAndVowelHarmony:
    """Rigorous linguistic tests for vowel harmony and consonant assimilation."""

    @pytest.mark.parametrize("word,expected_last_vowel,is_front", [
        ("өстәл", "ә", True),
        ("китап", "а", False),
        ("дәфтәр", "ә", True),
        ("урман", "а", False),
        ("күз", "ү", True),
        ("сыйныф", "ы", False),
        ("җәй", "ә", True),
        ("болыт", "ы", False),
        ("төн", "ө", True),
        ("таң", "а", False),
    ])
    def test_vowel_classification(self, word, expected_last_vowel, is_front):
        assert get_last_vowel(word) == expected_last_vowel
        assert is_front_vowel(word) is is_front

    @pytest.mark.parametrize("stem,case_code,expected_result", [
        # Voiceless consonant endings (п, к, т, с, ч, ш): -ка/-кә, -та/-тә, -тан/-тән
        ("китап", "case_dative", "КИТАПКА"),
        ("китап", "case_locative", "КИТАПТА"),
        ("китап", "case_ablative", "КИТАПТАН"),
        ("мәктәп", "case_dative", "МӘКТӘПКӘ"),
        ("мәктәп", "case_locative", "МӘКТӘПТӘ"),
        ("мәктәп", "case_ablative", "МӘКТӘПТӘН"),
        ("агач", "case_dative", "АГАЧКА"),
        ("агач", "case_locative", "АГАЧТА"),
        ("агач", "case_ablative", "АГАЧТАН"),
        ("кибет", "case_dative", "КИБЕТКӘ"),
        ("кибет", "case_locative", "КИБЕТТӘ"),
        ("кибет", "case_ablative", "КИБЕТТӘН"),
        
        # Nasal consonant endings (м, н, ң): ablative takes -нан/-нән, dative/locative take standard voiced
        ("урман", "case_ablative", "УРМАННАН"),
        ("урман", "case_dative", "УРМАНГА"),
        ("урман", "case_locative", "УРМАНДА"),
        ("урман", "case_genitive", "УРМАННЫҢ"),
        ("урман", "case_accusative", "УРМАННЫ"),
        ("төн", "case_ablative", "ТӨННӘН"),
        ("төн", "case_dative", "ТӨНГӘ"),
        ("төн", "case_locative", "ТӨНДӘ"),
        ("таң", "case_ablative", "ТАҢНАН"),
        ("урам", "case_ablative", "УРАМНАН"),

        # Vowels & voiced consonants: -га/-гә, -да/-дә, -дан/-дән
        ("бала", "case_dative", "БАЛАГА"),
        ("бала", "case_locative", "БАЛАДА"),
        ("бала", "case_ablative", "БАЛАДАН"),
        ("өстәл", "case_dative", "ӨСТӘЛГӘ"),
        ("өстәл", "case_locative", "ӨСТӘЛДӘ"),
        ("өстәл", "case_ablative", "ӨСТӘЛДӘН"),
        ("өстәл", "case_genitive", "ӨСТӘЛНЕҢ"),
        ("өстәл", "case_accusative", "ӨСТӘЛНЕ"),
        ("өй", "case_dative", "ӨЙГӘ"),
        ("өй", "case_locative", "ӨЙДӘ"),
        ("өй", "case_ablative", "ӨЙДӘН"),
    ])
    def test_case_inflections(self, stem, case_code, expected_result):
        assert inflect_case(stem, case_code) == expected_result

    @pytest.mark.parametrize("stem,expected_plural", [
        # Nasal assimilation (-нар / -нәр)
        ("урман", "УРМАННАР"),
        ("төн", "ТӨННӘР"),
        ("таң", "ТАҢНАР"),
        ("урам", "УРАМНАР"),
        ("көн", "КӨННӘР"),
        ("кием", "КИЕМНӘР"),
        
        # Standard harmony (-лар / -ләр)
        ("бала", "БАЛАЛАР"),
        ("өстәл", "ӨСТӘЛЛӘР"),
        ("китап", "КИТАПЛАР"),
        ("мәктәп", "МӘКТӘПЛӘР"),
        ("дәфтәр", "ДӘФТӘРЛӘР"),
        ("шәһәр", "ШӘҺӘРЛӘР"),
    ])
    def test_plural_inflections(self, stem, expected_plural):
        assert inflect_plural(stem) == expected_plural

    def test_cell_unicode_breakdown_tatar_letters(self):
        """Verifies special Tatar letters Ә, Ө, Ү, Җ, Ң, Һ map to exact Unicode hex."""
        tatar_word = "ӘҖҢӨҮҺ"
        cells = build_expected_cells(tatar_word, cell_count=8)
        
        assert len(cells) == 8
        assert cells[0] == {"index": 0, "char": "Ә", "unicode": "U+04D8", "is_empty_allowed": False}
        assert cells[1] == {"index": 1, "char": "Җ", "unicode": "U+0496", "is_empty_allowed": False}
        assert cells[2] == {"index": 2, "char": "Ң", "unicode": "U+04A2", "is_empty_allowed": False}
        assert cells[3] == {"index": 3, "char": "Ө", "unicode": "U+04E8", "is_empty_allowed": False}
        assert cells[4] == {"index": 4, "char": "Ү", "unicode": "U+04AE", "is_empty_allowed": False}
        assert cells[5] == {"index": 5, "char": "Һ", "unicode": "U+04BA", "is_empty_allowed": False}
        # Trailing cells
        assert cells[6] == {"index": 6, "char": " ", "unicode": "U+0020", "is_empty_allowed": True}
        assert cells[7] == {"index": 7, "char": " ", "unicode": "U+0020", "is_empty_allowed": True}

    def test_adversarial_and_invalid_inputs(self):
        """Checks robust error raising on non-Tatar, empty, or un-inflectable words."""
        with pytest.raises(PhoneticsError):
            clean_word("")
            
        with pytest.raises(PhoneticsError):
            clean_word("   ")
            
        with pytest.raises(PhoneticsError):
            clean_word("EnglishWord")
            
        with pytest.raises(PhoneticsError):
            clean_word("Слово123")
            
        with pytest.raises(PhoneticsError):
            get_last_vowel("бвгдж")
            
        with pytest.raises(PhoneticsError):
            get_case_suffix("китап", "invalid_case_code")
