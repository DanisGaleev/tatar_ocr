from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel

class FrequentWeakTopic(BaseModel):
    topic_code: str
    topic_name_tt: str
    total_questions: int
    wrong_count: int
    accuracy_pct: float

class ProblematicLetter(BaseModel):
    letter: str
    total_occurrences: int
    misrecognized_or_wrong: int
    accuracy_pct: float
    common_confusions: List[str]

class StudentHistoryItem(BaseModel):
    submission_id: str
    assignment_id: str
    assignment_title: str
    date: Optional[str] = None
    score: float
    max_score: float
    grade: int

class HandwritingMetrics(BaseModel):
    quality_pct: float
    status: str  # "EXCELLENT", "GOOD", "NEEDS_ATTENTION", "POOR"
    average_confidence: float
    teacher_correction_rate_pct: float
    low_confidence_rate_pct: float
    total_characters_analyzed: int
    unclear_characters: List[str]

class StudentAnalyticsResponse(BaseModel):
    student_id: str
    full_name: str
    class_name: str
    total_tests_completed: int
    average_score_pct: float
    average_grade: float
    grade_distribution: Dict[str, int]
    frequent_weak_topics: List[FrequentWeakTopic]
    problematic_letters: List[ProblematicLetter]
    history: List[StudentHistoryItem]
    handwriting_quality_pct: float = 100.0
    handwriting_status: str = "EXCELLENT"
    handwriting_metrics: Optional[HandwritingMetrics] = None

class TopClassMistake(BaseModel):
    topic_code: str
    topic_name_tt: str
    failure_rate_pct: float
    affected_students_count: int

class DifficultCharacter(BaseModel):
    letter: str
    error_rate_pct: float

class StudentPerformanceRow(BaseModel):
    student_id: str
    full_name: str
    average_score_pct: float
    average_grade: float
    tests_completed: int
    handwriting_quality_pct: float = 100.0
    handwriting_status: str = "EXCELLENT"

class ClassAnalyticsResponse(BaseModel):
    class_id: str
    class_name: str
    students_count: int
    average_class_score_pct: float
    average_handwriting_quality_pct: float = 100.0
    grade_distribution: Dict[str, int]
    top_class_mistakes: List[TopClassMistake]
    difficult_characters_across_class: List[DifficultCharacter]
    students_performance_table: List[StudentPerformanceRow]
    students_needing_handwriting_attention: List[str] = []

class WrongSubmissionItem(BaseModel):
    student_id: str
    student_name: str
    written_answer: str

class QuestionAnalyticsItem(BaseModel):
    question_number: int
    marker_id: int
    prompt: str
    expected_answer: str
    accuracy_pct: float
    wrong_submissions: List[WrongSubmissionItem]

class AssignmentAnalyticsResponse(BaseModel):
    assignment_id: str
    title: str
    total_submissions: int
    average_score_pct: float
    questions_analytics: List[QuestionAnalyticsItem]
