import json
import uuid
from typing import Optional, List
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Header, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.models.submission import SubmissionModel
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
                confidence=float(c.get("confidence", 1.0)),
                status=c.get("status", "MATCH"),
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
    inserted_count = 0
    updated_count = 0

    for item in payload.submissions:
        # Check if already present by client_submission_uuid
        sub_obj = None
        if item.client_submission_uuid:
            stmt = select(SubmissionModel).where(SubmissionModel.client_submission_uuid == item.client_submission_uuid)
            res = await db.execute(stmt)
            sub_obj = res.scalar_one_or_none()

        questions_json = json.dumps([q.model_dump() for q in item.questions_results], ensure_ascii=False)
        checked_time = item.checked_at or datetime.now(timezone.utc)

        if sub_obj:
            sub_obj.student_name = item.student_name
            sub_obj.variant = item.variant
            sub_obj.checked_at = checked_time
            sub_obj.overall_score = item.overall_score
            sub_obj.max_score = item.max_score
            sub_obj.final_grade = item.final_grade
            sub_obj.teacher_reviewed_flags = item.teacher_reviewed_flags
            sub_obj.questions_results_json = questions_json
            updated_count += 1
        else:
            new_sub = SubmissionModel(
                submission_id=f"sub_{uuid.uuid4().hex[:8]}",
                client_submission_uuid=item.client_submission_uuid or str(uuid.uuid4()),
                assignment_id=payload.assignment_id,
                class_id=payload.class_id,
                student_id=item.student_id,
                student_name=item.student_name,
                variant=item.variant,
                checked_at=checked_time,
                overall_score=item.overall_score,
                max_score=item.max_score,
                final_grade=item.final_grade,
                teacher_reviewed_flags=item.teacher_reviewed_flags,
                questions_results_json=questions_json,
            )
            db.add(new_sub)
            inserted_count += 1

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
    item = payload.submission
    questions_json = json.dumps([q.model_dump() for q in item.questions_results], ensure_ascii=False)
    checked_time = item.checked_at or datetime.now(timezone.utc)
    sub_id = f"sub_{uuid.uuid4().hex[:8]}"

    sub_obj = SubmissionModel(
        submission_id=sub_id,
        client_submission_uuid=item.client_submission_uuid or str(uuid.uuid4()),
        assignment_id=payload.assignment_id,
        class_id=payload.class_id,
        student_id=item.student_id,
        student_name=item.student_name,
        variant=item.variant,
        checked_at=checked_time,
        overall_score=item.overall_score,
        max_score=item.max_score,
        final_grade=item.final_grade,
        teacher_reviewed_flags=item.teacher_reviewed_flags,
        questions_results_json=questions_json,
    )
    db.add(sub_obj)
    await db.commit()

    return SingleSubmissionSaveResponse(status="saved", submission_id=sub_id)


@router.get(
    "",
    response_model=List[SubmissionItem],
    summary="Query Submissions",
)
async def query_submissions(
    class_id: Optional[str] = Query(None, description="Filter by class ID"),
    assignment_id: Optional[str] = Query(None, description="Filter by assignment ID"),
    student_id: Optional[str] = Query(None, description="Filter by student ID"),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(SubmissionModel)
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
