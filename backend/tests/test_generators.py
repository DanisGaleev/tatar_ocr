import pytest
import random
from app.generators.registry import registry
from app.generators.case_inflection import CaseInflectionGenerator
from app.generators.plural_affixes import PluralAffixesGenerator
from app.generators.antonyms import AntonymsGenerator
from app.generators.translation import TranslationGenerator

class TestTaskGeneratorsAndDeterminism:
    """Verifies deterministic reproducibility, random seed isolation, and generator accuracy."""

    def test_case_inflection_determinism(self):
        gen = CaseInflectionGenerator()
        draft_a = gen.generate(seed=42)
        draft_b = gen.generate(seed=42)
        draft_c = gen.generate(seed=999)

        # Exact match with identical seed
        assert draft_a.prompt_tt == draft_b.prompt_tt
        assert draft_a.expected_answer == draft_b.expected_answer
        assert draft_a.expected_cells == draft_b.expected_cells
        assert draft_a.metadata == draft_b.metadata

        # Different seed generates different item or case
        assert (draft_a.prompt_tt != draft_c.prompt_tt) or (draft_a.expected_answer != draft_c.expected_answer)

    def test_plural_affixes_determinism_and_rules(self):
        gen = PluralAffixesGenerator()
        draft_a = gen.generate(seed=777)
        draft_b = gen.generate(seed=777)

        assert draft_a.prompt_tt == draft_b.prompt_tt
        assert draft_a.expected_answer == draft_b.expected_answer
        assert len(draft_a.expected_answer) <= draft_a.cell_count <= 12

    def test_antonyms_generator_determinism(self):
        gen = AntonymsGenerator()
        draft_a = gen.generate(seed=101)
        draft_b = gen.generate(seed=101)

        assert draft_a.prompt_tt == draft_b.prompt_tt
        assert draft_a.expected_answer == draft_b.expected_answer
        assert "antonym" in draft_a.metadata

    def test_translation_generator_determinism(self):
        gen = TranslationGenerator()
        draft_a = gen.generate(seed=202)
        draft_b = gen.generate(seed=202)

        assert draft_a.prompt_tt == draft_b.prompt_tt
        assert draft_a.expected_answer == draft_b.expected_answer
        assert "ru_word" in draft_a.metadata

    def test_global_random_state_isolation(self):
        """Ensures passing a seed does not contaminate Python's global random generator."""
        random.seed(12345)
        expected_next_rand = random.random()
        
        # Reset and run our generator with a different seed
        random.seed(12345)
        gen = CaseInflectionGenerator()
        _ = gen.generate(seed=99999)
        
        # Global random stream must continue uninterrupted
        actual_next_rand = random.random()
        assert actual_next_rand == expected_next_rand

    def test_custom_task_creation_all_types(self):
        # 1. Custom case inflection
        case_gen = CaseInflectionGenerator()
        custom_case = case_gen.create_custom(stem="китап", case_code="case_locative")
        assert custom_case.expected_answer == "КИТАПТА"
        assert custom_case.cell_count == 8
        assert custom_case.metadata.get("is_custom") is True

        # 2. Custom plural with nasal assimilation
        plural_gen = PluralAffixesGenerator()
        custom_plural = plural_gen.create_custom(stem="урам")
        assert custom_plural.expected_answer == "УРАМНАР"
        assert custom_plural.metadata.get("is_nasal_rule") is True

        # 3. Custom antonym
        antonym_gen = AntonymsGenerator()
        custom_ant = antonym_gen.create_custom(word="кайнар", antonym="салкын")
        assert custom_ant.expected_answer == "САЛКЫН"

        # 4. Custom translation
        trans_gen = TranslationGenerator()
        custom_tr = trans_gen.create_custom(ru_word="ученик", tt_word="укучы")
        assert custom_tr.expected_answer == "УКУЧЫ"

    def test_batch_generation_respects_count_and_no_empty_fields(self):
        gen = CaseInflectionGenerator()
        batch = gen.generate_batch(count=5, seed=54321)
        
        assert len(batch) == 5
        for item in batch:
            assert item.prompt_tt
            assert item.expected_answer
            assert len(item.expected_answer) <= item.cell_count
            assert len(item.expected_cells) == item.cell_count

    def test_registry_assemble_modular_test_a4_limit(self):
        """Verifies assembled test creates valid A4 structure and rejects > 8 questions."""
        # Valid 8 questions
        bundle = registry.assemble_variants(
            assignment_id="TAT-TEST-01",
            title="7 сыйныф. Исем килешләре",
            grade_level=7,
            variants_count=2,
            seed=42,
            questions_per_variant=8,
        )
        assert bundle["assignment_id"] == "TAT-TEST-01"
        assert bundle["total_variants"] == 2
        assert len(bundle["variants"]) == 2
        
        var1 = bundle["variants"][0]
        assert len(var1["questions"]) == 8
        assert var1["template_geometry"]["format"] == "A4"
        assert var1["template_geometry"]["corner_aruco_ids"] == [0, 1, 2, 3]

        # Invariant: Question numbers must be 1..8 with sequential markers 11..18
        for idx, q in enumerate(var1["questions"], start=1):
            assert q["question_number"] == idx
            assert q["marker_id"] == 10 + idx

        # Physical limit violation (> 8 questions) must raise ValueError
        with pytest.raises(ValueError, match="exceeds A4 physical page limit"):
            registry.assemble_variants(
                assignment_id="TAT-OVERFLOW",
                title="Overflow Test",
                grade_level=7,
                questions_per_variant=9,
            )
