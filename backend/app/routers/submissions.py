import json
import uuid
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.teacher_scope import assert_class_access, normalize_teacher_uuid
from app.generators.checker import GradingScale, calculate_final_grade, verify_test_submission
from app.models.school import ClassModel, StudentModel
from app.models.submission import SubmissionModel
from app.models.teacher import TeacherModel
from app.models.test import AssembledTestModel
from app.schemas.submission import (
    BatchSyncSubmissionsRequest,
    BatchSyncSubmissionsResponse,
    SingleSubmissionSaveRequest,
    SingleSubmissionSaveResponse,
    SubmissionItem,
    SubmissionQuestionResult,
    SubmissionCellResult,
)

router = APIRouter(prefix="/submissions", tags=["Submissions Ingestion"])


def _normalize_utc_naive(dt: Optional[datetime]) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def parse_submission_item(sub: SubmissionModel) -> SubmissionItem:
    try:
        q_results = json.loads(sub.questions_results_json)
    except Exception:
        q_results = []

    parsed_questions = []
    for q in q_results:
        cells = [
            SubmissionCellResult(
                cell_index=c.get("cell_index", 0),
                expected_char=c.get("expected_char", ""),
                predicted_char=c.get("predicted_char", ""),
                confidence=1.0 if c.get("confidence") is None else float(c.get("confidence")),
                status=c.get("status") or "MATCH",
                teacher_override=c.get("teacher_override"),
            )
            for c in q.get("cells", [])
        ]
        parsed_questions.append(
            SubmissionQuestionResult(
                question_number=q.get("question_number", 1),
                marker_id=q.get("marker_id", 0),
                topic_tag=q.get("topic_tag", ""),
                is_correct=bool(q.get("is_correct", False)),
                points_earned=float(q.get("points_earned", 0.0)),
                cells=cells,
            )
        )

    return SubmissionItem(
        client_submission_uuid=sub.client_submission_uuid,
        student_id=sub.student_id,
        student_name=sub.student_name,
        variant=sub.variant,
        checked_at=sub.checked_at,
        overall_score=sub.overall_score,
        max_score=sub.max_score,
        final_grade=sub.final_grade,
        teacher_reviewed_flags=sub.teacher_reviewed_flags,
        questions_results=parsed_questions,
    )


async def _load_submission_context(
    db: AsyncSession,
    assignment_id: str,
    class_id: str,
    x_teacher_uuid: Optional[str],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, StudentModel], GradingScale]:
    test_obj = (
        await db.execute(select(AssembledTestModel).where(AssembledTestModel.test_id == assignment_id))
    ).scalar_one_or_none()
    if not test_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assignment '{assignment_id}' not found.",
        )

    try:
        bundle_dict = json.loads(test_obj.bundle_json)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Corrupted test bundle JSON for '{assignment_id}'.",
        ) from exc

    variants_by_id: Dict[int, Dict[str, Any]] = {
        int(v["variant_id"]): v
        for v in bundle_dict.get("variants", [])
        if v.get("variant_id") is not None
    }

    cls_obj = (
        await db.execute(select(ClassModel).where(ClassModel.class_id == class_id))
    ).scalar_one_or_none()
    if not cls_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Class '{class_id}' not found.",
        )

    norm_teacher_uuid = normalize_teacher_uuid(x_teacher_uuid)
    assert_class_access(cls_obj, norm_teacher_uuid)

    students = (
        await db.execute(select(StudentModel).where(StudentModel.class_id == class_id))
    ).scalars().all()
    student_map = {s.student_id: s for s in students}

    teacher_uuid = norm_teacher_uuid or cls_obj.teacher_uuid
    grading_scale = GradingScale()
    if teacher_uuid:
        teacher_obj = (
            await db.execute(select(TeacherModel).where(TeacherModel.teacher_uuid == teacher_uuid))
        ).scalar_one_or_none()
        if teacher_obj:
            grading_scale = GradingScale(
                grade_5_min_pct=teacher_obj.grade_5_min_pct,
                grade_4_min_pct=teacher_obj.grade_4_min_pct,
                grade_3_min_pct=teacher_obj.grade_3_min_pct,
            )

    return variants_by_id, student_map, grading_scale


def _recalculate_submission(
    item: SubmissionItem,
    variants_by_id: Dict[int, Dict[str, Any]],
    grading_scale: GradingScale,
) -> Tuple[float, float, int, int, str]:
    variant_dict = variants_by_id.get(item.variant)
    if not variant_dict:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Variant {item.variant} not found in assignment bundle.",
        )

    if item.questions_results:
        variant_questions = variant_dict.get("questions", [])
        sub_payload = [
            {
                "question_number": q.question_number,
                "cells": [c.model_dump() for c in q.cells],
            }
            for q in item.questions_results
        ]
        try:
            verified = verify_test_submission(variant_questions, sub_payload, grading_scale=grading_scale)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

        override_count = sum(
            1
            for q in verified.questions_results
            for c in q.cells
            if c.teacher_override is not None
        )
        reviewed_flags = override_count if override_count > 0 else item.teacher_reviewed_flags
        questions_json = json.dumps(
            [q.model_dump() for q in verified.questions_results],
            ensure_ascii=False,
        )
        return (
            verified.overall_score,
            verified.max_score,
            verified.final_grade,
            reviewed_flags,
            questions_json,
        )

    pct = round((item.overall_score / item.max_score * 100.0), 1)
    recomputed_grade = calculate_final_grade(pct, grading_scale)
    return (
        item.overall_score,
        item.max_score,
        recomputed_grade,
        item.teacher_reviewed_flags,
        "[]",
    )


async def _upsert_verified_submission(
    db: AsyncSession,
    assignment_id: str,
    class_id: str,
    item: SubmissionItem,
    variants_by_id: Dict[int, Dict[str, Any]],
    student_map: Dict[str, StudentModel],
    grading_scale: GradingScale,
    batch_by_uuid: Optional[Dict[str, SubmissionModel]] = None,
    batch_by_student: Optional[Dict[Tuple[str, str, str], SubmissionModel]] = None,
) -> Tuple[SubmissionModel, bool]:
    student_obj = student_map.get(item.student_id)
    if not student_obj:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Student '{item.student_id}' does not belong to class '{class_id}'.",
        )

    overall_score, max_score, final_grade, reviewed_flags, questions_json = _recalculate_submission(
        item, variants_by_id, grading_scale
    )
    checked_time = _normalize_utc_naive(item.checked_at)
    raw_uuid = (item.client_submission_uuid or "").strip()
    client_uuid = raw_uuid or str(uuid.uuid4())
    student_key = (assignment_id, class_id, item.student_id)

    sub_obj: Optional[SubmissionModel] = None
    if raw_uuid:
        if batch_by_uuid is not None and raw_uuid in batch_by_uuid:
            sub_obj = batch_by_uuid[raw_uuid]
        else:
            sub_obj = (
                await db.execute(
                    select(SubmissionModel).where(SubmissionModel.client_submission_uuid == raw_uuid)
                )
            ).scalar_one_or_none()

        if sub_obj and (sub_obj.assignment_id, sub_obj.class_id, sub_obj.student_id) != student_key:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"client_submission_uuid '{raw_uuid}' already belongs to another submission.",
            )

    if sub_obj is None:
        if batch_by_student is not None and student_key in batch_by_student:
            sub_obj = batch_by_student[student_key]
        else:
            sub_obj = (
                await db.execute(
                    select(SubmissionModel).where(
                        SubmissionModel.assignment_id == assignment_id,
                        SubmissionModel.class_id == class_id,
                        SubmissionModel.student_id == item.student_id,
                    )
                )
            ).scalar_one_or_none()

    if sub_obj is not None:
        if raw_uuid:
            sub_obj.client_submission_uuid = raw_uuid
        sub_obj.student_name = student_obj.full_name
        sub_obj.variant = item.variant
        sub_obj.checked_at = checked_time
        sub_obj.overall_score = overall_score
        sub_obj.max_score = max_score
        sub_obj.final_grade = final_grade
        sub_obj.teacher_reviewed_flags = reviewed_flags
        sub_obj.questions_results_json = questions_json
        is_new = False
    else:
        sub_obj = SubmissionModel(
            submission_id=f"sub_{uuid.uuid4().hex[:8]}",
            client_submission_uuid=client_uuid,
            assignment_id=assignment_id,
            class_id=class_id,
            student_id=item.student_id,
            student_name=student_obj.full_name,
            variant=item.variant,
            checked_at=checked_time,
            overall_score=overall_score,
            max_score=max_score,
            final_grade=final_grade,
            teacher_reviewed_flags=reviewed_flags,
            questions_results_json=questions_json,
        )
        db.add(sub_obj)
        is_new = True

    if batch_by_uuid is not None and sub_obj.client_submission_uuid:
        batch_by_uuid[sub_obj.client_submission_uuid] = sub_obj
    if batch_by_student is not None:
        batch_by_student[student_key] = sub_obj

    return sub_obj, is_new


@router.post(
    "/batch-sync",
    response_model=BatchSyncSubmissionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch Sync Graded Submissions (Zero-Photo Class Upload)",
)
async def batch_sync_submissions(
    payload: BatchSyncSubmissionsRequest,
    x_teacher_uuid: Optional[str] = Header(None, alias="X-Teacher-UUID"),
    db: AsyncSession = Depends(get_db),
):
    variants_by_id, student_map, grading_scale = await _load_submission_context(
        db, payload.assignment_id, payload.class_id, x_teacher_uuid
    )

    inserted_count = 0
    updated_count = 0
    batch_by_uuid: Dict[str, SubmissionModel] = {}
    batch_by_student: Dict[Tuple[str, str, str], SubmissionModel] = {}

    for item in payload.submissions:
        _, is_new = await _upsert_verified_submission(
            db=db,
            assignment_id=payload.assignment_id,
            class_id=payload.class_id,
            item=item,
            variants_by_id=variants_by_id,
            student_map=student_map,
            grading_scale=grading_scale,
            batch_by_uuid=batch_by_uuid,
            batch_by_student=batch_by_student,
        )
        if is_new:
            inserted_count += 1
        else:
            updated_count += 1

    await db.commit()

    return BatchSyncSubmissionsResponse(
        status="synced",
        received_count=len(payload.submissions),
        inserted_count=inserted_count,
        updated_count=updated_count,
        analytics_recalculated=True,
    )


@router.post(
    "",
    response_model=SingleSubmissionSaveResponse,
    status_code=status.HTTP_200_OK,
    summary="Save Single Graded Submission",
)
async def save_single_submission(
    payload: SingleSubmissionSaveRequest,
    x_teacher_uuid: Optional[str] = Header(None, alias="X-Teacher-UUID"),
    db: AsyncSession = Depends(get_db),
):
    variants_by_id, student_map, grading_scale = await _load_submission_context(
        db, payload.assignment_id, payload.class_id, x_teacher_uuid
    )

    sub_obj, _ = await _upsert_verified_submission(
        db=db,
        assignment_id=payload.assignment_id,
        class_id=payload.class_id,
        item=payload.submission,
        variants_by_id=variants_by_id,
        student_map=student_map,
        grading_scale=grading_scale,
    )
    await db.commit()

    return SingleSubmissionSaveResponse(status="saved", submission_id=sub_obj.submission_id)


@router.get(
    "",
    response_model=List[SubmissionItem],
    summary="Query Submissions",
)
async def query_submissions(
    class_id: Optional[str] = Query(None, description="Filter by class ID"),
    assignment_id: Optional[str] = Query(None, description="Filter by assignment ID"),
    student_id: Optional[str] = Query(None, description="Filter by student ID"),
    x_teacher_uuid: Optional[str] = Header(None, alias="X-Teacher-UUID"),
    db: AsyncSession = Depends(get_db),
):
    teacher_uuid = normalize_teacher_uuid(x_teacher_uuid)
    stmt = select(SubmissionModel)
    if teacher_uuid:
        if class_id:
            cls_obj = (
                await db.execute(select(ClassModel).where(ClassModel.class_id == class_id))
            ).scalar_one_or_none()
            if cls_obj:
                assert_class_access(cls_obj, teacher_uuid)
        stmt = stmt.join(ClassModel, SubmissionModel.class_id == ClassModel.class_id).where(
            (ClassModel.teacher_uuid == teacher_uuid) | (ClassModel.teacher_uuid.is_(None))
        )
    if class_id:
        stmt = stmt.where(SubmissionModel.class_id == class_id)
    if assignment_id:
        stmt = stmt.where(SubmissionModel.assignment_id == assignment_id)
    if student_id:
        stmt = stmt.where(SubmissionModel.student_id == student_id)

    stmt = stmt.order_by(SubmissionModel.checked_at.desc())
    res = await db.execute(stmt)
    submissions = res.scalars().all()

    return [parse_submission_item(s) for s in submissions]
