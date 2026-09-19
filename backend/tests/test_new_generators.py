import pytest
from app.generators.registry import registry
from app.generators.phonetics import (
    inflect_possessive,
    inflect_possessive_case,
    inflect_verb_negation,
    inflect_verb_imperative,
    inflect_comparative,
    inflect_ordinal_numeral,
    inflect_profession,
    inflect_abstract_noun,
    inflect_consonant_voicing,
    build_expected_cells,
)
from app.generators.morphology import (
    PossessiveAffixesGenerator,
    PossessiveCaseGenerator,
    VerbNegationGenerator,
    VerbImperativeGenerator,
    ComparativeDegreeGenerator,
    OrdinalNumeralsGenerator,
)
from app.generators.word_formation import (
    NounProfessionGenerator,
    AbstractNounGenerator,
    ConsonantVoicingGenerator,
)
from app.generators.lexicon_tasks import (
    SynonymsGenerator,
    CompoundWordsGenerator,
    OddOneOutGenerator,
)
from app.generators.phonetics_tasks import (
    VowelHarmonyGapGenerator,
    TatarLettersGapGenerator,
)
from app.generators.syntax_tasks import (
    SentenceMCClozeGenerator,
    TrueFalseGrammarGenerator,
    FindErrorSentenceGenerator,
    WordOrderGenerator,
)
from app.generators.reading_tasks import (
    TextComprehensionGenerator,
    TextTitleMainIdeaGenerator,
)
from app.generators.pdf_blank import render_blank_pdf


class TestTatarLinguisticInflections:
    """Rigorous linguistic tests for all Tatar grammar, phonetics, and orthography rules."""

    def test_possessive_inflection_all_persons_and_voicing(self):
        # 1. Back vowel stem with consonant voicing (китап: п -> б)
        assert inflect_possessive("китап", person=1) == "КИТАБЫМ"
        assert inflect_possessive("китап", person=2) == "КИТАБЫҢ"
        assert inflect_possessive("китап", person=3) == "КИТАБЫ"

        # 2. Back vowel stem with k -> g (тарак: к -> г)
        assert inflect_possessive("тарак", person=1) == "ТАРАГЫМ"
        assert inflect_possessive("тарак", person=2) == "ТАРАГЫҢ"
        assert inflect_possessive("тарак", person=3) == "ТАРАГЫ"

        # 3. Front vowel stem with consonant voicing (чиләк: к -> г)
        assert inflect_possessive("чиләк", person=1) == "ЧИЛӘГЕМ"
        assert inflect_possessive("чиләк", person=2) == "ЧИЛӘГЕҢ"
        assert inflect_possessive("чиләк", person=3) == "ЧИЛӘГЕ"

        # 4. Front vowel stem (өй)
        assert inflect_possessive("өй", person=1) == "ӨЕМ"
        assert inflect_possessive("өй", person=2) == "ӨЕҢ"
        assert inflect_possessive("өй", person=3) == "ӨЕ"

        # 5. Vowel-ending stem (бала)
        assert inflect_possessive("бала", person=1) == "БАЛАМ"
        assert inflect_possessive("бала", person=2) == "БАЛАҢ"
        assert inflect_possessive("бала", person=3) == "БАЛАСЫ"

        # 6. Word ending in 'и' (әни)
        assert inflect_possessive("әни", person=1) == "ӘНИЕМ"
        assert inflect_possessive("әни", person=2) == "ӘНИЕҢ"
        assert inflect_possessive("әни", person=3) == "ӘНИСЕ"

    def test_possessive_case_intersection_and_pronominal_n(self):
        """
        Crucial Tatar test: 3rd person possessive inserts pronominal 'н'
        before ANY case suffix!
        """
        # 3rd person + accusative: китабы + н -> КИТАБЫН (not китабыны)
        assert inflect_possessive_case("китап", person=3, case_code="case_accusative") == "КИТАБЫН"
        # 3rd person + dative: китабы + н + а -> КИТАБЫНА (not китабыга)
        assert inflect_possessive_case("китап", person=3, case_code="case_dative") == "КИТАБЫНА"
        # 3rd person + locative: китабы + н + да -> КИТАБЫНДА (not китабыда)
        assert inflect_possessive_case("китап", person=3, case_code="case_locative") == "КИТАБЫНДА"
        # 3rd person + ablative: китабы + н + нан -> КИТАБЫННАН (not китабыдан)
        assert inflect_possessive_case("китап", person=3, case_code="case_ablative") == "КИТАБЫННАН"
        # 3rd person + front locative: өе + н + дә -> ӨЕНДӘ
        assert inflect_possessive_case("өй", person=3, case_code="case_locative") == "ӨЕНДӘ"

        # 1st person possessive + cases
        # өем + дә -> ӨЕМДӘ
        assert inflect_possessive_case("өй", person=1, case_code="case_locative") == "ӨЕМДӘ"
        # өем + нән -> ӨЕМНӘН (nasal assimilation after м)
        assert inflect_possessive_case("өй", person=1, case_code="case_ablative") == "ӨЕМНӘН"
        # өем + ә -> ӨЕМӘ (literary dative)
        assert inflect_possessive_case("өй", person=1, case_code="case_dative") == "ӨЕМӘ"

    def test_verb_negation(self):
        # Back vowels: -мый
        assert inflect_verb_negation("бар") == "БАРМЫЙ"
        assert inflect_verb_negation("яз") == "ЯЗМЫЙ"
        assert inflect_verb_negation("укы") == "УКЫМЫЙ"
        assert inflect_verb_negation("кара") == "КАРАМЫЙ"
        # Front vowels: -ми
        assert inflect_verb_negation("кил") == "КИЛМИ"
        assert inflect_verb_negation("бел") == "БЕЛМИ"
        assert inflect_verb_negation("эшлә") == "ЭШЛӘМИ"
        assert inflect_verb_negation("сөйлә") == "СӨЙЛӘМИ"

    def test_verb_imperative(self):
        # Consonant stems: -ыгыз / -егез
        assert inflect_verb_imperative("бар") == "БАРЫГЫЗ"
        assert inflect_verb_imperative("яз") == "ЯЗЫГЫЗ"
        assert inflect_verb_imperative("кил") == "КИЛЕГЕЗ"
        assert inflect_verb_imperative("бел") == "БЕЛЕГЕЗ"
        assert inflect_verb_imperative("тап") == "ТАБЫГЫЗ"  # п -> б voicing!
        # Vowel stems: -гыз / -гез
        assert inflect_verb_imperative("укы") == "УКЫГЫЗ"
        assert inflect_verb_imperative("кара") == "КАРАГЫЗ"
        assert inflect_verb_imperative("эшлә") == "ЭШЛӘГЕЗ"
        assert inflect_verb_imperative("сөйлә") == "СӨЙЛӘГЕЗ"

    def test_comparative_degree(self):
        # Back: -рак
        assert inflect_comparative("зур") == "ЗУРРАК"
        assert inflect_comparative("салкын") == "САЛКЫНРАК"
        assert inflect_comparative("матур") == "МАТУРРАК"
        assert inflect_comparative("озын") == "ОЗЫНРАК"
        # Front: -рәк
        assert inflect_comparative("тиз") == "ТИЗРӘК"
        assert inflect_comparative("яшел") == "ЯШЕЛРӘК"
        assert inflect_comparative("җиңел") == "ҖИҢЕЛРӘК"

    def test_ordinal_numerals(self):
        # Consonant stems: -ынчы / -енче
        assert inflect_ordinal_numeral("бер") == "БЕРЕНЧЕ"
        assert inflect_ordinal_numeral("өч") == "ӨЧЕНЧЕ"
        assert inflect_ordinal_numeral("дүрт") == "ДҮРТЕНЧЕ"
        assert inflect_ordinal_numeral("биш") == "БИШЕНЧЕ"
        assert inflect_ordinal_numeral("сигез") == "СИГЕЗЕНЧЕ"
        assert inflect_ordinal_numeral("тугыз") == "ТУГЫЗЫНЧЫ"
        assert inflect_ordinal_numeral("ун") == "УНЫНЧЫ"
        assert inflect_ordinal_numeral("кырык") == "КЫРЫГЫНЧЫ"  # к -> г voicing!
        # Vowel stems: -нчы / -нче
        assert inflect_ordinal_numeral("ике") == "ИКЕНЧЕ"
        assert inflect_ordinal_numeral("алты") == "АЛТЫНЧЫ"
        assert inflect_ordinal_numeral("җиде") == "ҖИДЕНЧЕ"
        assert inflect_ordinal_numeral("егерме") == "ЕГЕРМЕНЧЕ"

    def test_word_formation_inflections(self):
        # Profession: -чы / -че
        assert inflect_profession("балык") == "БАЛЫКЧЫ"
        assert inflect_profession("эш") == "ЭШЧЕ"
        assert inflect_profession("җыр") == "ҖЫРЧЫ"
        assert inflect_profession("укыту") == "УКЫТУЧЫ"

        # Abstract nouns: -лык / -лек
        assert inflect_abstract_noun("матур") == "МАТУРЛЫК"
        assert inflect_abstract_noun("иркен") == "ИРКЕНЛЕК"
        assert inflect_abstract_noun("дус") == "ДУСЛЫК"
        assert inflect_abstract_noun("яшь") == "ЯШЬЛЕК"

        # Consonant voicing
        assert inflect_consonant_voicing("тарак") == "ТАРАГЫ"
        assert inflect_consonant_voicing("китап") == "КИТАБЫ"
        assert inflect_consonant_voicing("чиләк") == "ЧИЛӘГЕ"


class TestAllNewGenerators:
    """Verifies that each new generator class produces valid, deterministic, constrained tasks."""

    @pytest.mark.parametrize("gen_cls", [
        PossessiveAffixesGenerator,
        PossessiveCaseGenerator,
        VerbNegationGenerator,
        VerbImperativeGenerator,
        ComparativeDegreeGenerator,
        OrdinalNumeralsGenerator,
        NounProfessionGenerator,
        AbstractNounGenerator,
        ConsonantVoicingGenerator,
        SynonymsGenerator,
        CompoundWordsGenerator,
        OddOneOutGenerator,
        VowelHarmonyGapGenerator,
        TatarLettersGapGenerator,
        SentenceMCClozeGenerator,
        TrueFalseGrammarGenerator,
        FindErrorSentenceGenerator,
        WordOrderGenerator,
        TextComprehensionGenerator,
        TextTitleMainIdeaGenerator,
    ])
    def test_generator_determinism_and_cell_bounds(self, gen_cls):
        gen = gen_cls()
        draft_a = gen.generate(seed=42)
        draft_b = gen.generate(seed=42)

        # 1. Determinism
        assert draft_a.prompt_tt == draft_b.prompt_tt
        assert draft_a.expected_answer == draft_b.expected_answer
        assert draft_a.cell_count == draft_b.cell_count
        assert draft_a.expected_cells == draft_b.expected_cells

        # 2. Cell count constraints
        assert 1 <= draft_a.cell_count <= 12
        assert len(draft_a.expected_answer) <= draft_a.cell_count

        # 3. Expected cells structure
        assert len(draft_a.expected_cells) == draft_a.cell_count
        for idx, cell in enumerate(draft_a.expected_cells):
            assert cell["index"] == idx
            assert "char" in cell
            assert "unicode" in cell
            assert cell["unicode"].startswith("U+")

    def test_single_cell_generators(self):
        """Phonetic gaps, Multiple Choice, and True/False must strictly produce 1-cell answers."""
        single_cell_gens = [
            VowelHarmonyGapGenerator(),
            TatarLettersGapGenerator(),
            SentenceMCClozeGenerator(),
            TrueFalseGrammarGenerator(),
            FindErrorSentenceGenerator(),
            TextComprehensionGenerator(),
            TextTitleMainIdeaGenerator(),
        ]
        for gen in single_cell_gens:
            draft = gen.generate(seed=123)
            assert draft.cell_count == 1
            assert len(draft.expected_answer) == 1
            assert len(draft.expected_cells) == 1


class TestRegistryAndPDFRendering:
    """Verifies registration and PDF blank generation for all new task types."""

    def test_registry_contains_all_24_types(self):
        types = [item["task_type"] for item in registry.list_supported_types()]
        expected_types = [
            "case_inflection", "plural_affixes", "antonyms", "translation",
            "possessive_affixes", "possessive_case", "verb_negation", "verb_imperative",
            "comparative_degree", "ordinal_numerals", "noun_profession", "abstract_noun",
            "consonant_voicing", "synonyms", "compound_words", "odd_one_out",
            "vowel_harmony_gap", "tatar_letters_gap", "sentence_mc_cloze", "true_false_grammar",
            "find_error_sentence", "word_order", "text_comprehension", "text_title_main_idea"
        ]
        for exp in expected_types:
            assert exp in types
            gen = registry.get_generator(exp)
            assert gen is not None

    def test_pdf_blank_renders_with_multiline_and_single_cell_questions(self):
        """Verifies that PDF blank rendering succeeds with multi-line and single-cell items."""
        questions = [
            {
                "question_number": 1,
                "marker_id": 11,
                "prompt": "«Мин китап ___ укыйм»\nДөрес кушымчаны сайлагыз: А) ны   Б) ка   В) дан",
                "cell_count": 1,
                "expected_answer": "А",
            },
            {
                "question_number": 2,
                "marker_id": 12,
                "prompt": "Казанда яз җитте. Кояш җылыта, карлар эри.\nКошлар каян кайталар? А) Җылы яклардан   Б) Урманнан   В) Күлдән",
                "cell_count": 1,
                "expected_answer": "А",
            },
            {
                "question_number": 3,
                "marker_id": 13,
                "prompt": "Тартым кушымчасын ялгагыз (I зат): китап ->",
                "cell_count": 8,
                "expected_answer": "КИТАБЫМ",
            },
            {
                "question_number": 4,
                "marker_id": 14,
                "prompt": "Җөмлә төзегез (хәрефләр тәртибен языгыз): А) мәктәпкә   Б) Мин   В) барам   Г) һәрвакыт",
                "cell_count": 4,
                "expected_answer": "БГАВ",
            }
        ]

        pdf_bytes = render_blank_pdf(
            assignment_id="TEST-MULTI-2026",
            title="Катнаш контроль эш (Синтаксис һәм Морфология)",
            variant_id=1,
            questions=questions,
            student_name="Галиев Илдар",
        )

        assert pdf_bytes.startswith(b"%PDF")
        assert len(pdf_bytes) > 50_000
