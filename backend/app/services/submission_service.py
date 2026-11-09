from datetime import datetime, timezone
import json
from typing import Dict, Any, List, Optional
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.core.telemetry import telemetry_manager
from app.models.user import UserModel
from app.models.submission import SubmissionModel
from app.repositories.submission_repo import SubmissionRepository
from app.repositories.class_repo import ClassRepository
from app.repositories.audit_repo import AuditRepository
from app.schemas.submission import (
    SingleSubmissionSaveRequest,
    SingleSubmissionSaveResponse,
    BatchSyncSubmissionsRequest,
    BatchSyncSubmissionsResponse,
    SubmissionItem,
    SubmissionQuestionResult,
    SubmissionCellResult,
)


class SubmissionService:
    """Service layer managing submission ingestion, batch-sync, grade persistence, and querying."""

    def __init__(self, db: AsyncSession):
        self._db: AsyncSession = db
        self._submission_repo = SubmissionRepository(db)
        self._class_repo = ClassRepository(db)
        self._audit_repo = AuditRepository(db)

    async def save_single_submission(
        self,
        payload: SingleSubmissionSaveRequest,
        current_user: UserModel,
    ) -> SingleSubmissionSaveResponse:
        await self._verify_class_access_if_present(payload.class_id, current_user)
        item = payload.submission
        data = {
            "assignment_id": payload.assignment_id,
            "class_id": payload.class_id,
            "client_submission_uuid": item.client_submission_uuid or str(uuid.uuid4()),
            "student_id": item.student_id,
            "student_name": item.student_name,
            "variant": item.variant,
            "checked_at": item.checked_at or datetime.now(timezone.utc),
            "overall_score": item.overall_score,
            "max_score": item.max_score,
            "final_grade": item.final_grade,
            "teacher_reviewed_flags": item.teacher_reviewed_flags,
            "questions_results": item.questions_results,
            "teacher_id": current_user.id,
        }
        sub_obj, _ = await self._submission_repo.save_or_update(submission_data=data)
        return SingleSubmissionSaveResponse(status="saved", submission_id=sub_obj.submission_id)

    async def batch_sync_submissions(
        self,
        payload: BatchSyncSubmissionsRequest,
        current_user: UserModel,
    ) -> BatchSyncSubmissionsResponse:
        cls_obj = await self._verify_class_access_if_present(payload.class_id, current_user)
        submissions_data = []
        for item in payload.submissions:
            submissions_data.append({
                "assignment_id": payload.assignment_id,
                "class_id": payload.class_id,
                "client_submission_uuid": item.client_submission_uuid or str(uuid.uuid4()),
                "student_id": item.student_id,
                "student_name": item.student_name,
                "variant": item.variant,
                "checked_at": item.checked_at or datetime.now(timezone.utc),
                "overall_score": item.overall_score,
                "max_score": item.max_score,
                "final_grade": item.final_grade,
                "teacher_reviewed_flags": item.teacher_reviewed_flags,
                "questions_results": item.questions_results,
                "teacher_id": current_user.id,
            })

        inserted, updated = await self._submission_repo.batch_create(submissions_data)

        # Broadcast telemetry event
        await telemetry_manager.broadcast({
            "action": "SUBMISSION_SYNC",
            "user_id": current_user.id,
            "user_name": current_user.full_name,
            "school": current_user.school_name or "",
            "details": {
                "class_id": payload.class_id,
                "class_name": cls_obj.name if cls_obj else payload.class_id,
                "assignment_id": payload.assignment_id,
                "submissions_count": len(payload.submissions),
                "inserted": inserted,
                "updated": updated,
            },
        })

        return BatchSyncSubmissionsResponse(
            status="synced",
            received_count=len(payload.submissions),
            inserted_count=inserted,
            updated_count=updated,
            analytics_recalculated=True,
        )

    async def list_submissions(
        self,
        *,
        class_id: Optional[str] = None,
        assignment_id: Optional[str] = None,
        student_id: Optional[str] = None,
        current_user: UserModel,
    ) -> List[SubmissionItem]:
        if class_id:
            await self._verify_class_access_if_present(class_id, current_user)

        is_admin = current_user.role in ("admin", "superadmin")
        submissions = await self._submission_repo.query_submissions(
            class_id=class_id,
            assignment_id=assignment_id,
            student_id=student_id,
            teacher_id=current_user.id,
            is_admin=is_admin,
        )
        return [self._parse_submission_item(s) for s in submissions]

    async def get_submission(self, submission_id: str, current_user: UserModel) -> SubmissionItem:
        sub = await self._submission_repo.get_by_id(submission_id)
        if not sub:
            raise NotFoundError(message=f"Submission '{submission_id}' not found.")

        if current_user.role not in ("admin", "superadmin"):
            if sub.teacher_id and sub.teacher_id != current_user.id:
                raise PermissionDeniedError(message="Access denied: You do not own this submission")
            if sub.class_id:
                cls_obj = await self._class_repo.get_by_id(sub.class_id)
                if cls_obj and cls_obj.teacher_uuid and cls_obj.teacher_uuid != current_user.id:
                    raise PermissionDeniedError(message="Access denied: You do not own this submission")

        return self._parse_submission_item(sub)

    async def clear_student_submissions(
        self,
        *,
        student_id: str,
        class_id: Optional[str] = None,
        current_user: UserModel,
    ) -> int:
        if class_id:
            await self._verify_class_access_if_present(class_id, current_user)
        return await self._submission_repo.clear_student_submissions(student_id=student_id, class_id=class_id)

    # -------------------------------------------------------------------------
    # Internal Helpers (Marked Private with Leading Underscore)
    # -------------------------------------------------------------------------
    async def _verify_class_access_if_present(
        self,
        class_id: Optional[str],
        current_user: UserModel,
    ) -> Optional[Any]:
        if not class_id:
            return None
        cls_obj = await self._class_repo.get_by_id(class_id)
        if cls_obj and cls_obj.teacher_uuid and current_user.role not in ("admin", "superadmin"):
            if cls_obj.teacher_uuid != current_user.id:
                raise PermissionDeniedError(
                    message="Access denied: You do not own this class",
                    details={"class_id": class_id},
                )
        return cls_obj

    def _parse_submission_item(self, sub: SubmissionModel) -> SubmissionItem:
        try:
            q_results = json.loads(sub.questions_results_json) if sub.questions_results_json else []
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
