import json
from collections import defaultdict
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.models.school import ClassModel, StudentModel
from app.models.submission import SubmissionModel
from app.models.test import AssembledTestModel
from app.schemas.analytics import (
    StudentAnalyticsResponse,
    FrequentWeakTopic,
    ProblematicLetter,
    StudentHistoryItem,
    ClassAnalyticsResponse,
    TopClassMistake,
    DifficultCharacter,
    StudentPerformanceRow,
    AssignmentAnalyticsResponse,
    QuestionAnalyticsItem,
    WrongSubmissionItem,
)

router = APIRouter(prefix="/analytics", tags=["Backend Analytics Engine"])

# Topic code to human Tatar display name mapping
TOPIC_NAMES = {
    "case_dative": "Юнәлеш килеше (-ка/-кә, -га/-гә)",
    "case_ablative": "Чыгыш килеше (-дан/-дән, -тан/-тән, -нан/-нән)",
    "case_locative": "Урын-вакыт килеше (-да/-дә, -та/-тә)",
    "case_accusative": "Төшем килеше (-ны/-не)",
    "case_genitive": "Иялек килеше (-ның/-нең)",
    "case_inflection": "Исем килешләре",
    "plural_affixes": "Күплек сан кушымчалары (-лар/-ләр, -нар/-нәр)",
    "antonyms": "Сыйфат антонимнары (кире мәгънә)",
    "antonyms_adjectives": "Сыйфат антонимнары (кире мәгънә)",
    "translation": "Сүзлек байлыгы (русча-татарча тәрҗемә)",
    "vocabulary_translation": "Сүзлек байлыгы (русча-татарча тәрҗемә)",
}


@router.get("/students/{student_id}", response_model=StudentAnalyticsResponse, summary="Student Analytics (Individual)")
async def get_student_analytics(
    student_id: str,
    x_teacher_uuid: Optional[str] = Header(None, alias="X-Teacher-UUID"),
    db: AsyncSession = Depends(get_db),
):
    stmt_stu = select(StudentModel).where(StudentModel.student_id == student_id)
    res_stu = await db.execute(stmt_stu)
    student = res_stu.scalar_one_or_none()

    student_name = student.full_name if student else f"Укучы {student_id}"
    class_name = "Билгесез"
    if student:
        stmt_cls = select(ClassModel).where(ClassModel.class_id == student.class_id)
        res_cls = await db.execute(stmt_cls)
        cls_obj = res_cls.scalar_one_or_none()
        if cls_obj:
            class_name = cls_obj.name

    stmt_sub = select(SubmissionModel).where(SubmissionModel.student_id == student_id).order_by(SubmissionModel.checked_at.desc())
    res_sub = await db.execute(stmt_sub)
    submissions = res_sub.scalars().all()

    if not submissions:
        return StudentAnalyticsResponse(
            student_id=student_id,
            full_name=student_name,
            class_name=class_name,
            total_tests_completed=0,
            average_score_pct=0.0,
            average_grade=0.0,
            grade_distribution={"5": 0, "4": 0, "3": 0, "2": 0},
            frequent_weak_topics=[],
            problematic_letters=[],
            history=[],
        )

    total_tests = len(submissions)
    pcts = []
    grades = []
    grade_dist = {"5": 0, "4": 0, "3": 0, "2": 0}
    history = []

    topic_stats = defaultdict(lambda: {"total": 0, "wrong": 0})
    letter_stats = defaultdict(lambda: {"total": 0, "wrong": 0, "confusions": set()})

    for s in submissions:
        pct = (s.overall_score / s.max_score * 100.0) if s.max_score > 0 else 0.0
        pcts.append(pct)
        grades.append(s.final_grade)
        g_str = str(s.final_grade)
        if g_str in grade_dist:
            grade_dist[g_str] += 1

        date_str = s.checked_at.strftime("%Y-%m-%dT%H:%M:%SZ") if s.checked_at else None
        history.append(
            StudentHistoryItem(
                submission_id=s.submission_id,
                assignment_id=s.assignment_id,
                assignment_title=s.assignment_id,
                date=date_str,
                score=s.overall_score,
                max_score=s.max_score,
                grade=s.final_grade,
            )
        )

        try:
            q_results = json.loads(s.questions_results_json)
        except Exception:
            q_results = []

        for q in q_results:
            tag = q.get("topic_tag", "general")
            topic_stats[tag]["total"] += 1
            if not q.get("is_correct", False):
                topic_stats[tag]["wrong"] += 1

            for c in q.get("cells", []):
                exp = c.get("expected_char", "").strip().upper()
                pred = c.get("predicted_char", "").strip().upper()
                status_cell = c.get("status", "MATCH")
                override = c.get("teacher_override")

                if exp and exp.isalpha():
                    letter_stats[exp]["total"] += 1
                    is_wrong = (status_cell != "MATCH") or (pred != exp) or (override == "WRONG")
                    if is_wrong:
                        letter_stats[exp]["wrong"] += 1
                        if pred and pred != exp and pred.isalpha():
                            letter_stats[exp]["confusions"].add(pred)

    avg_score_pct = round(sum(pcts) / total_tests, 1) if total_tests else 0.0
    avg_grade = round(sum(grades) / total_tests, 1) if total_tests else 0.0

    weak_topics = []
    for code, st in topic_stats.items():
        acc = round((st["total"] - st["wrong"]) / st["total"] * 100.0, 1) if st["total"] > 0 else 100.0
        weak_topics.append(
            FrequentWeakTopic(
                topic_code=code,
                topic_name_tt=TOPIC_NAMES.get(code, code),
                total_questions=st["total"],
                wrong_count=st["wrong"],
                accuracy_pct=acc,
            )
        )
    weak_topics.sort(key=lambda x: (x.accuracy_pct, -x.wrong_count))

    prob_letters = []
    for ltr, st in letter_stats.items():
        if st["wrong"] > 0:
            acc = round((st["total"] - st["wrong"]) / st["total"] * 100.0, 1) if st["total"] > 0 else 100.0
            prob_letters.append(
                ProblematicLetter(
                    letter=ltr,
                    total_occurrences=st["total"],
                    misrecognized_or_wrong=st["wrong"],
                    accuracy_pct=acc,
                    common_confusions=sorted(list(st["confusions"]))[:3],
                )
            )
    prob_letters.sort(key=lambda x: (x.accuracy_pct, -x.misrecognized_or_wrong))

    return StudentAnalyticsResponse(
        student_id=student_id,
        full_name=student_name,
        class_name=class_name,
        total_tests_completed=total_tests,
        average_score_pct=avg_score_pct,
        average_grade=avg_grade,
        grade_distribution=grade_dist,
        frequent_weak_topics=weak_topics,
        problematic_letters=prob_letters,
        history=history,
    )


@router.get("/classes/{class_id}", response_model=ClassAnalyticsResponse, summary="Class Aggregated Analytics & Heatmap")
async def get_class_analytics(
    class_id: str,
    x_teacher_uuid: Optional[str] = Header(None, alias="X-Teacher-UUID"),
    db: AsyncSession = Depends(get_db),
):
    stmt_cls = select(ClassModel).where(ClassModel.class_id == class_id)
    res_cls = await db.execute(stmt_cls)
    cls_obj = res_cls.scalar_one_or_none()

    class_name = cls_obj.name if cls_obj else class_id

    stmt_stu = select(StudentModel).where(StudentModel.class_id == class_id)
    res_stu = await db.execute(stmt_stu)
    all_students = res_stu.scalars().all()
    students_count = len(all_students)

    stmt_sub = select(SubmissionModel).where(SubmissionModel.class_id == class_id)
    res_sub = await db.execute(stmt_sub)
    submissions = res_sub.scalars().all()

    if not submissions:
        return ClassAnalyticsResponse(
            class_id=class_id,
            class_name=class_name,
            students_count=students_count,
            average_class_score_pct=0.0,
            grade_distribution={"5": 0, "4": 0, "3": 0, "2": 0},
            top_class_mistakes=[],
            difficult_characters_across_class=[],
            students_performance_table=[
                StudentPerformanceRow(
                    student_id=s.student_id,
                    full_name=s.full_name,
                    average_score_pct=0.0,
                    average_grade=0.0,
                    tests_completed=0,
                )
                for s in all_students
            ],
        )

    pcts = []
    grade_dist = {"5": 0, "4": 0, "3": 0, "2": 0}
    student_records = defaultdict(lambda: {"pcts": [], "grades": [], "name": ""})
    topic_mistakes = defaultdict(lambda: {"total": 0, "wrong": 0, "affected_students": set()})
    char_errors = defaultdict(lambda: {"total": 0, "wrong": 0})

    for s in submissions:
        pct = (s.overall_score / s.max_score * 100.0) if s.max_score > 0 else 0.0
        pcts.append(pct)
        g_str = str(s.final_grade)
        if g_str in grade_dist:
            grade_dist[g_str] += 1

        student_records[s.student_id]["pcts"].append(pct)
        student_records[s.student_id]["grades"].append(s.final_grade)
        student_records[s.student_id]["name"] = s.student_name

        try:
            q_results = json.loads(s.questions_results_json)
        except Exception:
            q_results = []

        for q in q_results:
            tag = q.get("topic_tag", "general")
            topic_mistakes[tag]["total"] += 1
            if not q.get("is_correct", False):
                topic_mistakes[tag]["wrong"] += 1
                topic_mistakes[tag]["affected_students"].add(s.student_id)

            for c in q.get("cells", []):
                exp = c.get("expected_char", "").strip().upper()
                pred = c.get("predicted_char", "").strip().upper()
                status_cell = c.get("status", "MATCH")
                override = c.get("teacher_override")

                if exp and exp.isalpha():
                    char_errors[exp]["total"] += 1
                    is_wrong = (status_cell != "MATCH") or (pred != exp) or (override == "WRONG")
                    if is_wrong:
                        char_errors[exp]["wrong"] += 1

    avg_class_score_pct = round(sum(pcts) / len(pcts), 1) if pcts else 0.0

    top_mistakes = []
    for tag, tm in topic_mistakes.items():
        fail_pct = round(tm["wrong"] / tm["total"] * 100.0, 1) if tm["total"] > 0 else 0.0
        top_mistakes.append(
            TopClassMistake(
                topic_code=tag,
                topic_name_tt=TOPIC_NAMES.get(tag, tag),
                failure_rate_pct=fail_pct,
                affected_students_count=len(tm["affected_students"]),
            )
        )
    top_mistakes.sort(key=lambda x: -x.failure_rate_pct)

    difficult_chars = []
    for ch, ce in char_errors.items():
        if ce["wrong"] > 0:
            err_rate = round(ce["wrong"] / ce["total"] * 100.0, 1) if ce["total"] > 0 else 0.0
            difficult_chars.append(DifficultCharacter(letter=ch, error_rate_pct=err_rate))
    difficult_chars.sort(key=lambda x: -x.error_rate_pct)

    perf_table = []
    # Include all known students
    student_ids_set = {s.student_id for s in all_students} | set(student_records.keys())
    name_map = {s.student_id: s.full_name for s in all_students}

    for stu_id in student_ids_set:
        rec = student_records.get(stu_id, {"pcts": [], "grades": [], "name": ""})
        s_pcts = rec["pcts"]
        s_grades = rec["grades"]
        full_n = name_map.get(stu_id) or rec["name"] or f"Укучы {stu_id}"

        avg_pct = round(sum(s_pcts) / len(s_pcts), 1) if s_pcts else 0.0
        avg_g = round(sum(s_grades) / len(s_grades), 1) if s_grades else 0.0
        perf_table.append(
            StudentPerformanceRow(
                student_id=stu_id,
                full_name=full_n,
                average_score_pct=avg_pct,
                average_grade=avg_g,
                tests_completed=len(s_pcts),
            )
        )
    perf_table.sort(key=lambda x: (-x.average_score_pct, -x.average_grade, x.full_name))

    return ClassAnalyticsResponse(
        class_id=class_id,
        class_name=class_name,
        students_count=max(students_count, len(perf_table)),
        average_class_score_pct=avg_class_score_pct,
        grade_distribution=grade_dist,
        top_class_mistakes=top_mistakes,
        difficult_characters_across_class=difficult_chars,
        students_performance_table=perf_table,
    )


@router.get("/assignments/{assignment_id}", response_model=AssignmentAnalyticsResponse, summary="Assignment Analytics (Item Analysis)")
async def get_assignment_analytics(
    assignment_id: str,
    x_teacher_uuid: Optional[str] = Header(None, alias="X-Teacher-UUID"),
    db: AsyncSession = Depends(get_db),
):
    stmt_test = select(AssembledTestModel).where(AssembledTestModel.test_id == assignment_id)
    res_test = await db.execute(stmt_test)
    test_obj = res_test.scalar_one_or_none()

    title = test_obj.title if test_obj else f"Контроль эш {assignment_id}"

    # Extract questions metadata from bundle if available
    bundle_questions = {}
    if test_obj:
        try:
            b_dict = json.loads(test_obj.bundle_json)
            for v in b_dict.get("variants", []):
                for q in v.get("questions", []):
                    q_num = q.get("question_number")
                    if q_num not in bundle_questions:
                        bundle_questions[q_num] = {
                            "marker_id": q.get("marker_id", 10 + q_num),
                            "prompt": q.get("prompt", ""),
                            "expected_answer": q.get("expected_answer", ""),
                        }
        except Exception:
            pass

    stmt_sub = select(SubmissionModel).where(SubmissionModel.assignment_id == assignment_id)
    res_sub = await db.execute(stmt_sub)
    submissions = res_sub.scalars().all()

    total_submissions = len(submissions)
    if not submissions:
        return AssignmentAnalyticsResponse(
            assignment_id=assignment_id,
            title=title,
            total_submissions=0,
            average_score_pct=0.0,
            questions_analytics=[],
        )

    pcts = []
    q_stats = defaultdict(lambda: {"total": 0, "correct": 0, "marker_id": 0, "prompt": "", "expected_answer": "", "wrong": []})

    for s in submissions:
        pct = (s.overall_score / s.max_score * 100.0) if s.max_score > 0 else 0.0
        pcts.append(pct)

        try:
            q_results = json.loads(s.questions_results_json)
        except Exception:
            q_results = []

        for q in q_results:
            q_num = q.get("question_number", 1)
            b_info = bundle_questions.get(q_num, {})
            m_id = q.get("marker_id") or b_info.get("marker_id", 10 + q_num)
            p_text = b_info.get("prompt", f"Сорау №{q_num}")
            e_ans = b_info.get("expected_answer", "")

            q_stats[q_num]["marker_id"] = m_id
            q_stats[q_num]["prompt"] = p_text
            q_stats[q_num]["expected_answer"] = e_ans
            q_stats[q_num]["total"] += 1

            is_corr = bool(q.get("is_correct", False))
            if is_corr:
                q_stats[q_num]["correct"] += 1
            else:
                # Reconstruct student written answer from cells
                chars = [c.get("predicted_char", "") for c in q.get("cells", [])]
                written = "".join(chars).strip()
                q_stats[q_num]["wrong"].append(
                    WrongSubmissionItem(
                        student_id=s.student_id,
                        student_name=s.student_name,
                        written_answer=written or "[Буш / Пусто]",
                    )
                )

    avg_score_pct = round(sum(pcts) / total_submissions, 1)

    questions_out = []
    for q_num in sorted(q_stats.keys()):
        stat = q_stats[q_num]
        acc_pct = round(stat["correct"] / stat["total"] * 100.0, 1) if stat["total"] > 0 else 0.0
        questions_out.append(
            QuestionAnalyticsItem(
                question_number=q_num,
                marker_id=stat["marker_id"],
                prompt=stat["prompt"],
                expected_answer=stat["expected_answer"],
                accuracy_pct=acc_pct,
                wrong_submissions=stat["wrong"],
            )
        )

    return AssignmentAnalyticsResponse(
        assignment_id=assignment_id,
        title=title,
        total_submissions=total_submissions,
        average_score_pct=avg_score_pct,
        questions_analytics=questions_out,
    )
