import unicodedata
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Mapping of Latin lookalikes to Cyrillic homoglyphs to eliminate OCR encoding discrepancies
HOMOGLYPH_MAP = {
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н',
    'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р', 'T': 'Т',
    'X': 'Х', 'Y': 'Ү',
}

RUSSIAN_HOMOGLYPH_MAP = {
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н',
    'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р', 'T': 'Т',
    'X': 'Х', 'Y': 'У',
    'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р',
    'x': 'х', 'y': 'у'
}

COMMON_TATAR_CONFUSIONS = {
    'Ң': ['Н'],
    'Ә': ['А'],
    'Ө': ['О'],
    'Ү': ['У'],
    'Җ': ['Ж'],
    'Һ': ['Х'],
}

def normalize_char(char: Optional[str], language: str = "tt") -> str:
    if not char:
        return " "
    c = unicodedata.normalize("NFC", char).strip()
    if not c:
        return " "
    upper_c = c.upper()
    mapping = RUSSIAN_HOMOGLYPH_MAP if language == "ru" else HOMOGLYPH_MAP
    return mapping.get(upper_c, upper_c)

def normalize_answer(text: str, language: str = "tt") -> str:
    clean = unicodedata.normalize("NFC", text or "").strip()
    return "".join(normalize_char(ch, language=language) for ch in clean)

class CellVerificationResult(BaseModel):
    cell_index: int
    expected_char: str
    predicted_char: str
    confidence: float = 1.0
    status: str  # MATCH, MISMATCH, EMPTY_MISMATCH, EMPTY_MATCH, FLAG_OVERRIDDEN_BY_TEACHER
    teacher_override: Optional[str] = None  # CORRECT, WRONG

class QuestionVerificationResult(BaseModel):
    question_number: int
    marker_id: int
    topic_tag: str
    is_correct: bool
    points_earned: float
    max_points: float = 1.0
    cells: List[CellVerificationResult]
    written_word: str

class GradingScale(BaseModel):
    grade_5_min_pct: int = 85
    grade_4_min_pct: int = 70
    grade_3_min_pct: int = 50

class TestVerificationResult(BaseModel):
    overall_score: float
    max_score: float
    score_percentage: float
    final_grade: int
    questions_results: List[QuestionVerificationResult]

def verify_question_cells(
    question_number: int,
    marker_id: int,
    topic_tag: str,
    expected_answer: str,
    cell_count: int,
    student_cells: List[Dict[str, Any]],
    language: str = "tt",
) -> QuestionVerificationResult:
    """
    Evaluates student handwritten character recognitions against the expected answer.
    Handles trailing empty cells, homoglyphs, and teacher overrides.
    """
    if cell_count < 1:
        raise ValueError("cell_count must be >= 1")

    clean_expected = normalize_answer(expected_answer, language=language)
    if not clean_expected:
        raise ValueError("expected_answer cannot be empty")
    if len(clean_expected) > cell_count:
        raise ValueError(
            f"expected_answer length ({len(clean_expected)}) exceeds cell_count ({cell_count})"
        )

    cell_results: List[CellVerificationResult] = []
    has_extra_student_chars = False
    student_map: Dict[int, Dict[str, Any]] = {}
    for idx, c in enumerate(student_cells):
        c_idx = c.get("cell_index", idx)
        if c_idx < 0 or c_idx >= cell_count:
            if normalize_char(c.get("predicted_char", " "), language=language) != " ":
                has_extra_student_chars = True
        else:
            student_map[c_idx] = c

    all_matched = True
    written_chars: List[str] = []

    for idx in range(cell_count):
        exp_char = clean_expected[idx] if idx < len(clean_expected) else " "
        is_empty_allowed = (idx >= len(clean_expected))

        st_data = student_map.get(idx, {})
        pred_raw = st_data.get("predicted_char", " ")
        pred_norm = normalize_char(pred_raw, language=language)
        raw_conf = st_data.get("confidence")
        confidence = 1.0 if raw_conf is None else float(raw_conf)
        raw_override = st_data.get("teacher_override")
        norm_override = (
            raw_override.strip().upper()
            if isinstance(raw_override, str) and raw_override.strip()
            else None
        )
        override = norm_override if norm_override in ("CORRECT", "WRONG") else None
        written_chars.append(pred_norm)

        if override == "CORRECT":
            status = "FLAG_OVERRIDDEN_BY_TEACHER"
            matched = True
        elif override == "WRONG":
            status = "FLAG_OVERRIDDEN_BY_TEACHER"
            matched = False
        else:
            if is_empty_allowed:
                if pred_norm == " ":
                    status = "EMPTY_MATCH"
                    matched = True
                else:
                    status = "MISMATCH"
                    matched = False
            else:
                if pred_norm == exp_char:
                    status = "MATCH"
                    matched = True
                elif pred_norm == " ":
                    status = "EMPTY_MISMATCH"
                    matched = False
                else:
                    status = "MISMATCH"
                    matched = False

        if not matched:
            all_matched = False

        cell_results.append(CellVerificationResult(
            cell_index=idx,
            expected_char=exp_char,
            predicted_char=pred_norm,
            confidence=confidence,
            status=status,
            teacher_override=override,
        ))

    if has_extra_student_chars:
        all_matched = False

    points = 1.0 if all_matched else 0.0
    return QuestionVerificationResult(
        question_number=question_number,
        marker_id=marker_id,
        topic_tag=topic_tag,
        is_correct=all_matched,
        points_earned=points,
        max_points=1.0,
        cells=cell_results,
        written_word="".join(written_chars).strip(),
    )

def calculate_final_grade(percentage: float, scale: GradingScale) -> int:
    if percentage >= scale.grade_5_min_pct:
        return 5
    elif percentage >= scale.grade_4_min_pct:
        return 4
    elif percentage >= scale.grade_3_min_pct:
        return 3
    else:
        return 2

def verify_test_submission(
    questions: List[Dict[str, Any]],
    submissions: List[Dict[str, Any]],
    grading_scale: Optional[GradingScale] = None,
    language: str = "tt",
) -> TestVerificationResult:
    """
    Verifies an entire test sheet submission and computes total score and school grade (2-5).
    """
    if not questions:
        raise ValueError("questions cannot be empty")

    scale = grading_scale or GradingScale()
    q_results: List[QuestionVerificationResult] = []

    sub_map = {s.get("question_number"): s for s in submissions}

    total_score = 0.0
    max_score = float(len(questions))

    for q in questions:
        q_num = q["question_number"]
        marker_id = q.get("marker_id", 10 + q_num)
        topic_tag = q.get("topic_tag", "general")
        expected_answer = q["expected_answer"]
        cell_count = q.get("cell_count") or max(1, len(normalize_answer(expected_answer, language=language)))

        st_sub = sub_map.get(q_num, {})
        student_cells = st_sub.get("cells", [])

        res = verify_question_cells(
            question_number=q_num,
            marker_id=marker_id,
            topic_tag=topic_tag,
            expected_answer=expected_answer,
            cell_count=cell_count,
            student_cells=student_cells,
            language=language,
        )
        total_score += res.points_earned
        q_results.append(res)

    pct = round((total_score / max_score * 100.0), 1) if max_score > 0 else 0.0
    grade = calculate_final_grade(pct, scale)

    return TestVerificationResult(
        overall_score=total_score,
        max_score=max_score,
        score_percentage=pct,
        final_grade=grade,
        questions_results=q_results,
    )
