import re
import random
import pytest

from app.generators.phonetics import PhoneticsError
from app.generators.checker import normalize_char, verify_question_cells, verify_test_submission, GradingScale
from app.generators.pdf_blank import render_blank_pdf
from app.generators.registry import registry
from app.generators.russian import (
    RUSSIAN_UPPERCASE_33,
    TATAR_ONLY_LETTERS,
    clean_russian_word,
    mask_orthogram_combination,
    analyze_russian_phonetics,
    RuOrthogramCombinationsGenerator,
    RuUnstressedVowelGenerator,
    RuPairedConsonantGenerator,
    RuUnpronounceableConsonantGenerator,
    RuSoftHardSignGenerator,
    RuDictionaryWordsGenerator,
    RuNounGenderGenerator,
    RuNounNumberGenerator,
    RuPartsOfSpeechGenerator,
    RuNounCaseGenerator,
    RuAntonymsGenerator,
    RuSynonymsGenerator,
    RuPhoneticsGenerator,
    RuOddOneOutGenerator,
)
from app.generators.russian.lexicon import (
    ORTHOGRAM_COMBINATION_WORDS,
    UNSTRESSED_VOWEL_ITEMS,
    PAIRED_CONSONANT_ITEMS,
    UNPRONOUNCEABLE_CONSONANT_ITEMS,
    SOFT_HARD_SIGN_ITEMS,
    DICTIONARY_WORDS_ITEMS,
    NOUN_GENDER_ITEMS,
    NOUN_NUMBER_ITEMS,
    PARTS_OF_SPEECH_ITEMS,
    NOUN_CASE_ITEMS,
    RU_ANTONYM_PAIRS,
    RU_SYNONYM_SETS,
    PHONETICS_WORD_POOL,
    ODD_ONE_OUT_GROUPS,
)

RU_UPPER_PATTERN = re.compile(r"^[А-ЯЁ]+$")


class TestRussianLexiconIntegrity:
    """Verify every curated primary-school dataset item fits 1-12 cells and has zero Tatar letters."""

    def test_all_expected_answers_strictly_russian_and_within_cell_limit(self):
        for w in ORTHOGRAM_COMBINATION_WORDS:
            cleaned = clean_russian_word(w).upper()
            assert RU_UPPER_PATTERN.match(cleaned)
            assert 1 <= len(cleaned) <= 12
            masked, _ = mask_orthogram_combination(w)
            assert "_" in masked

        for word_gap, check_word, missing, full_word, grade_min in UNSTRESSED_VOWEL_ITEMS:
            assert 1 <= grade_min <= 4
            assert len(missing) == 1 and RU_UPPER_PATTERN.match(missing)
            assert word_gap.replace("_", missing.lower()) == full_word.lower()
            assert check_word

        for word_gap, check_word, missing, full_word, grade_min in PAIRED_CONSONANT_ITEMS:
            assert 1 <= grade_min <= 4
            assert len(missing) == 1 and RU_UPPER_PATTERN.match(missing)
            assert word_gap.replace("_", missing.lower()) == full_word.lower()
            assert check_word

        for word_gap, check_word, missing, full_word in UNPRONOUNCEABLE_CONSONANT_ITEMS:
            assert len(missing) == 1 and RU_UPPER_PATTERN.match(missing)
            assert word_gap.replace("_", missing.lower()) == full_word.lower()
            assert check_word

        for word_gap, sign, full_word in SOFT_HARD_SIGN_ITEMS:
            assert sign in ("Ъ", "Ь")
            assert word_gap.replace("_", sign.lower()) == full_word.lower()

        for word_gap, full_word, grade in DICTIONARY_WORDS_ITEMS:
            assert 1 <= grade <= 4
            assert RU_UPPER_PATTERN.match(full_word)
            assert len(full_word) <= 12
            assert word_gap.count("_") == 1

        for word, gender in NOUN_GENDER_ITEMS:
            clean_russian_word(word)
            assert gender in ("М", "Ж", "С")

        for singular, plural in NOUN_NUMBER_ITEMS:
            clean_russian_word(singular)
            assert RU_UPPER_PATTERN.match(plural)
            assert len(plural) <= 12

        for word, pos in PARTS_OF_SPEECH_ITEMS:
            clean_russian_word(word)
            assert pos in ("С", "П", "Г")

        for phrase, case_code in NOUN_CASE_ITEMS:
            assert case_code in ("И", "Р", "Д", "В", "Т", "П")
            assert any(ch not in TATAR_ONLY_LETTERS for ch in phrase)

        for word, ant in RU_ANTONYM_PAIRS:
            clean_russian_word(word)
            assert RU_UPPER_PATTERN.match(ant)
            assert len(ant) <= 12

        for word, corr, dists in RU_SYNONYM_SETS:
            clean_russian_word(word)
            clean_russian_word(corr)
            assert len(dists) >= 2
            for d in dists:
                clean_russian_word(d)

        for w in PHONETICS_WORD_POOL:
            clean_russian_word(w)

        for cat_words in ODD_ONE_OUT_GROUPS.values():
            assert len(cat_words) >= 3
            for w in cat_words:
                clean_russian_word(w)


class TestRussianPhoneticsHelpers:
    def test_clean_russian_word_rejects_tatar_latin_digits(self):
        assert clean_russian_word("  Ёжик ") == "Ёжик"
        for bad in ("китап", "әни", "өстәл", "җил", "көз", "үлән", "һава", "book", "123", ""):
            if bad in ("китап",):
                # китап uses shared Cyrillic letters so it is valid Cyrillic string; Tatar-only letters must fail:
                continue
            with pytest.raises(PhoneticsError):
                clean_russian_word(bad)
        for tatar_only in ("ә", "ө", "ү", "җ", "ң", "һ", "ӘНИ", "ӨЙ"):
            with pytest.raises(PhoneticsError):
                clean_russian_word(tatar_only)

    def test_mask_orthogram_combination(self):
        masked, rule = mask_orthogram_combination("лыжи")
        assert masked == "лыж_"
        assert rule == "ЖИ–ШИ"
        masked2, rule2 = mask_orthogram_combination("чашка")
        assert masked2 == "ч_шка"
        assert rule2 == "ЧА–ЩА"
        masked3, rule3 = mask_orthogram_combination("чувство")
        assert masked3 == "ч_вство"
        assert rule3 == "ЧУ–ЩУ"

    def test_analyze_russian_phonetics_rules(self):
        # ёжик: ё(2)+ж(1)+и(1)+к(1) = 5 sounds, 4 letters
        a1 = analyze_russian_phonetics("ёжик")
        assert a1["letter_count"] == 4
        assert a1["sound_count"] == 5

        # конь: к(1)+о(1)+н(1)+ь(0) = 3 sounds, 4 letters
        a2 = analyze_russian_phonetics("конь")
        assert a2["letter_count"] == 4
        assert a2["sound_count"] == 3

        # дом: д(hard), м(hard) -> all_consonants_hard
        a3 = analyze_russian_phonetics("дом")
        assert a3["all_consonants_hard"] is True
        assert a3["all_consonants_soft"] is False

        # Iotated vowels at start / after vowel / after ъ,ь emit soft [й'], so consonants are NOT all hard
        for iot_word in ("ёжик", "яблоко", "юла", "маяк", "заяц"):
            ai = analyze_russian_phonetics(iot_word)
            assert ai["all_consonants_hard"] is False, f"{iot_word} contains soft [й'] and cannot be all_consonants_hard"

        # мяч: м'(soft before я), ч'(always soft) -> all_consonants_soft
        a4 = analyze_russian_phonetics("мяч")
        assert a4["all_consonants_soft"] is True
        assert a4["all_consonants_hard"] is False


ALL_RUSSIAN_GENERATORS = [
    RuOrthogramCombinationsGenerator,
    RuUnstressedVowelGenerator,
    RuPairedConsonantGenerator,
    RuUnpronounceableConsonantGenerator,
    RuSoftHardSignGenerator,
    RuDictionaryWordsGenerator,
    RuNounGenderGenerator,
    RuNounNumberGenerator,
    RuPartsOfSpeechGenerator,
    RuNounCaseGenerator,
    RuAntonymsGenerator,
    RuSynonymsGenerator,
    RuPhoneticsGenerator,
    RuOddOneOutGenerator,
]


class TestAllRussianGenerators:
    @pytest.mark.parametrize("gen_cls", ALL_RUSSIAN_GENERATORS)
    def test_determinism_constraints_and_isolation(self, gen_cls):
        gen = gen_cls()
        assert gen.language == "ru"
        assert gen.category in {"orthography", "morphology", "lexicon", "phonetics"}

        random.seed(98765)
        expected_rand = random.random()

        random.seed(98765)
        draft_a = gen.generate(seed=42, grade_level=3)
        draft_b = gen.generate(seed=42, grade_level=3)
        assert random.random() == expected_rand

        assert draft_a.language == "ru"
        assert draft_a.prompt_tt == draft_b.prompt_tt
        assert draft_a.expected_answer == draft_b.expected_answer
        assert draft_a.cell_count == draft_b.cell_count
        assert draft_a.expected_cells == draft_b.expected_cells

        assert 1 <= draft_a.cell_count <= 12
        assert 1 <= len(draft_a.expected_answer) <= draft_a.cell_count
        assert RU_UPPER_PATTERN.match(draft_a.expected_answer)
        assert not any(ch in TATAR_ONLY_LETTERS for ch in draft_a.expected_answer)
        assert not any(ch in TATAR_ONLY_LETTERS for ch in draft_a.prompt_tt)
        assert len(draft_a.expected_cells) == draft_a.cell_count

        batch = gen.generate_batch(count=4, seed=1234, grade_level=3)
        assert len(batch) == 4
        assert all(d.language == "ru" for d in batch)

    def test_registry_has_all_14_russian_generators(self):
        ru_types = registry.list_supported_types(language="ru")
        tt_types = registry.list_supported_types(language="tt")
        assert len(ru_types) == 14
        assert len(tt_types) == 24
        assert {t["task_type"] for t in ru_types}.isdisjoint({t["task_type"] for t in tt_types})

    def test_create_custom_russian_generators(self):
        d1 = RuOrthogramCombinationsGenerator().create_custom(word="снежинка", grade_level=2)
        assert d1.expected_answer == "СНЕЖИНКА"
        assert "снеж_нка" in d1.prompt_tt

        d2 = RuNounGenderGenerator().create_custom(word="яблоко", gender_code="С", grade_level=3)
        assert d2.expected_answer == "С"

        d3 = RuNounCaseGenerator().create_custom(phrase="читает КНИГУ дома", case_code="В", grade_level=4)
        assert d3.expected_answer == "В"

        d4 = RuAntonymsGenerator().create_custom(word="свет", antonym="тьма", grade_level=2)
        assert d4.expected_answer == "ТЬМА"


class TestRussianCheckerAndPDF:
    def test_russian_homoglyph_normalization_maps_y_to_cyrillic_u(self):
        # Crucial difference: in Tatar Y -> Ү, in Russian Y -> У
        assert normalize_char("Y", language="tt") == "Ү"
        assert normalize_char("Y", language="ru") == "У"

        res = verify_question_cells(
            question_number=1,
            marker_id=11,
            topic_tag="ru_antonyms",
            expected_answer="УТРО",
            cell_count=8,
            student_cells=[
                {"cell_index": 0, "predicted_char": "Y"},
                {"cell_index": 1, "predicted_char": "T"},
                {"cell_index": 2, "predicted_char": "P"},
                {"cell_index": 3, "predicted_char": "O"},
            ],
            language="ru",
        )
        assert res.is_correct is True
        assert res.written_word == "УТРО"

    def test_russian_pdf_blank_renders_cleanly(self):
        bundle = registry.assemble_variants(
            assignment_id="RUS-TEST-2026",
            title="Русский язык. 3 класс. Проверочная работа",
            grade_level=3,
            variants_count=1,
            seed=42,
            questions_per_variant=8,
            language="ru",
        )
        assert bundle["language"] == "ru"
        assert bundle["alphabet_mask"] == "russian_33"
        questions = bundle["variants"][0]["questions"]
        assert len(questions) == 8

        pdf_bytes = render_blank_pdf(
            assignment_id="RUS-TEST-2026",
            title=bundle["title"],
            variant_id=1,
            questions=questions,
            student_name="Иванов Иван",
            language="ru",
        )
        assert pdf_bytes.startswith(b"%PDF")
        assert len(pdf_bytes) > 40_000
