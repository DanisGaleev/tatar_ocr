import pytest
from app.generators.checker import (
    verify_question_cells,
    verify_test_submission,
    calculate_final_grade,
    normalize_char,
    GradingScale,
)

class TestVerificationAndCheckingMechanism:
    """Rigorous tests for OCR answer checking, scoring, homoglyphs, and grading logic."""

    def test_exact_answer_match(self):
        expected = "КИТАПКА"  # 7 letters
        cell_count = 8
        # Student wrote all 7 letters correctly, 8th is empty
        student_cells = [
            {"cell_index": 0, "predicted_char": "К"},
            {"cell_index": 1, "predicted_char": "И"},
            {"cell_index": 2, "predicted_char": "Т"},
            {"cell_index": 3, "predicted_char": "А"},
            {"cell_index": 4, "predicted_char": "П"},
            {"cell_index": 5, "predicted_char": "К"},
            {"cell_index": 6, "predicted_char": "А"},
            {"cell_index": 7, "predicted_char": " "},
        ]
        res = verify_question_cells(
            question_number=1,
            marker_id=11,
            topic_tag="case_dative",
            expected_answer=expected,
            cell_count=cell_count,
            student_cells=student_cells,
        )
        assert res.is_correct is True
        assert res.points_earned == 1.0
        assert res.written_word == "КИТАПКА"
        assert res.cells[0].status == "MATCH"
        assert res.cells[7].status == "EMPTY_MATCH"

    def test_single_character_phonetic_error(self):
        """Student made a voicing error: written КИТАПГА instead of КИТАПКА."""
        expected = "КИТАПКА"
        student_cells = [
            {"cell_index": 0, "predicted_char": "К"},
            {"cell_index": 1, "predicted_char": "И"},
            {"cell_index": 2, "predicted_char": "Т"},
            {"cell_index": 3, "predicted_char": "А"},
            {"cell_index": 4, "predicted_char": "П"},
            {"cell_index": 5, "predicted_char": "Г"},  # Wrong character
            {"cell_index": 6, "predicted_char": "А"},
            {"cell_index": 7, "predicted_char": " "},
        ]
        res = verify_question_cells(
            question_number=1,
            marker_id=11,
            topic_tag="case_dative",
            expected_answer=expected,
            cell_count=8,
            student_cells=student_cells,
        )
        assert res.is_correct is False
        assert res.points_earned == 0.0
        assert res.cells[5].status == "MISMATCH"
        assert res.cells[5].expected_char == "К"
        assert res.cells[5].predicted_char == "Г"

    def test_empty_letter_mismatch(self):
        """Student skipped a required letter."""
        expected = "ӨСТӘЛДӘН"
        student_cells = [
            {"cell_index": 0, "predicted_char": "Ө"},
            {"cell_index": 1, "predicted_char": " "},  # Skipped letter
            {"cell_index": 2, "predicted_char": "Т"},
            {"cell_index": 3, "predicted_char": "Ә"},
            {"cell_index": 4, "predicted_char": "Л"},
            {"cell_index": 5, "predicted_char": "Д"},
            {"cell_index": 6, "predicted_char": "Ә"},
            {"cell_index": 7, "predicted_char": "Н"},
        ]
        res = verify_question_cells(
            question_number=2,
            marker_id=12,
            topic_tag="case_ablative",
            expected_answer=expected,
            cell_count=8,
            student_cells=student_cells,
        )
        assert res.is_correct is False
        assert res.cells[1].status == "EMPTY_MISMATCH"

    def test_extra_trailing_character_rejected(self):
        """Student wrote extra garbage character into trailing padding cell."""
        expected = "ХАКЛЫК"  # 6 letters
        student_cells = [
            {"cell_index": 0, "predicted_char": "Х"},
            {"cell_index": 1, "predicted_char": "А"},
            {"cell_index": 2, "predicted_char": "К"},
            {"cell_index": 3, "predicted_char": "Л"},
            {"cell_index": 4, "predicted_char": "Ы"},
            {"cell_index": 5, "predicted_char": "К"},
            {"cell_index": 6, "predicted_char": "А"},  # Extra character in empty slot
            {"cell_index": 7, "predicted_char": " "},
        ]
        res = verify_question_cells(
            question_number=3,
            marker_id=13,
            topic_tag="antonyms_adjectives",
            expected_answer=expected,
            cell_count=8,
            student_cells=student_cells,
        )
        assert res.is_correct is False
        assert res.cells[6].status == "MISMATCH"

    def test_latin_lookalike_homoglyph_normalization(self):
        """Verifies OCR outputting Latin 'A', 'C', 'O', 'P', 'X', 'E' matches Cyrillic."""
        assert normalize_char("A") == "А"  # Latin A -> Cyrillic А
        assert normalize_char("c") == "С"  # Latin c -> Cyrillic С
        assert normalize_char("O") == "О"  # Latin O -> Cyrillic О
        assert normalize_char("p") == "Р"  # Latin p -> Cyrillic Р
        assert normalize_char("x") == "Х"  # Latin x -> Cyrillic Х

        # Test verification with Latin characters
        expected = "АК"  # Cyrillic А, К
        student_cells = [
            {"cell_index": 0, "predicted_char": "A"},  # Latin A
            {"cell_index": 1, "predicted_char": "K"},  # Latin K
        ]
        res = verify_question_cells(
            question_number=1,
            marker_id=11,
            topic_tag="antonyms",
            expected_answer=expected,
            cell_count=2,
            student_cells=student_cells,
        )
        assert res.is_correct is True
        assert res.cells[0].status == "MATCH"

    def test_teacher_hitl_override(self):
        """Teacher manually overrides an ambiguous or flagged prediction."""
        expected = "РӘХМӘТ"
        student_cells = [
            {"cell_index": 0, "predicted_char": "Р"},
            # Model confused handwritten Ә with А, teacher overrides to CORRECT
            {"cell_index": 1, "predicted_char": "А", "teacher_override": "CORRECT"},
            {"cell_index": 2, "predicted_char": "Х"},
            {"cell_index": 3, "predicted_char": "М"},
            {"cell_index": 4, "predicted_char": "Ә"},
            {"cell_index": 5, "predicted_char": "Т"},
            {"cell_index": 6, "predicted_char": " "},
        ]
        res = verify_question_cells(
            question_number=4,
            marker_id=14,
            topic_tag="vocabulary_translation",
            expected_answer=expected,
            cell_count=7,
            student_cells=student_cells,
        )
        assert res.is_correct is True
        assert res.points_earned == 1.0
        assert res.cells[1].status == "FLAG_OVERRIDDEN_BY_TEACHER"

    def test_full_test_submission_and_grade_calculation(self):
        scale = GradingScale(grade_5_min_pct=85, grade_4_min_pct=70, grade_3_min_pct=50)
        
        assert calculate_final_grade(90.0, scale) == 5
        assert calculate_final_grade(85.0, scale) == 5
        assert calculate_final_grade(75.0, scale) == 4
        assert calculate_final_grade(70.0, scale) == 4
        assert calculate_final_grade(60.0, scale) == 3
        assert calculate_final_grade(50.0, scale) == 3
        assert calculate_final_grade(49.9, scale) == 2

        questions = [
            {"question_number": 1, "expected_answer": "КИТАПКА", "cell_count": 8},
            {"question_number": 2, "expected_answer": "ӨСТӘЛДӘН", "cell_count": 8},
            {"question_number": 3, "expected_answer": "БАЛАЛАР", "cell_count": 8},
            {"question_number": 4, "expected_answer": "ХАКЛЫК", "cell_count": 6},
        ]
        # Student answers 3 correctly out of 4 (75% -> Grade 4)
        submissions = [
            {"question_number": 1, "cells": [{"cell_index": i, "predicted_char": c} for i, c in enumerate("КИТАПКА ")]},
            {"question_number": 2, "cells": [{"cell_index": i, "predicted_char": c} for i, c in enumerate("ӨСТӘЛДӘН")]},
            {"question_number": 3, "cells": [{"cell_index": i, "predicted_char": c} for i, c in enumerate("БАЛАЛАР ")]},
            {"question_number": 4, "cells": [{"cell_index": i, "predicted_char": c} for i, c in enumerate("ЯЛГАН  ")]}, # Wrong
        ]
        result = verify_test_submission(questions, submissions, scale)
        assert result.overall_score == 3.0
        assert result.max_score == 4.0
        assert result.score_percentage == 75.0
        assert result.final_grade == 4
