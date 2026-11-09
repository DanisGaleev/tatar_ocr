import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func

from app.models.submission import SubmissionModel
from app.repositories.base import BaseRepository


class SubmissionRepository(BaseRepository[SubmissionModel]):
    """Data access repository for SubmissionModel."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, SubmissionModel)

    async def get_by_id(self, submission_id: str) -> Optional[SubmissionModel]:
        """Fetch submission by primary key."""
        return await self._session.get(SubmissionModel, submission_id)

    async def get_by_client_uuid(
        self,
        client_submission_uuid: str,
    ) -> Optional[SubmissionModel]:
        """Fetch submission by client idempotency UUID."""
        stmt = select(SubmissionModel).where(
            SubmissionModel.client_submission_uuid == client_submission_uuid
        )
        res = await self._session.execute(stmt)
        return res.scalar_one_or_none()

    async def list_by_class(self, class_id: str) -> List[SubmissionModel]:
        """List all submissions in class ordered by checked_at descending."""
        stmt = (
            select(SubmissionModel)
            .where(SubmissionModel.class_id == class_id)
            .order_by(SubmissionModel.checked_at.desc())
        )
        res = await self._session.execute(stmt)
        return list(res.scalars().all())

    async def list_by_student(self, student_id: str) -> List[SubmissionModel]:
        """List submissions for a specific student."""
        stmt = (
            select(SubmissionModel)
            .where(SubmissionModel.student_id == student_id)
            .order_by(SubmissionModel.checked_at.desc())
        )
        res = await self._session.execute(stmt)
        return list(res.scalars().all())

    async def list_by_assignment(self, assignment_id: str) -> List[SubmissionModel]:
        """List submissions for a specific assignment."""
        stmt = (
            select(SubmissionModel)
            .where(SubmissionModel.assignment_id == assignment_id)
            .order_by(SubmissionModel.checked_at.desc())
        )
        res = await self._session.execute(stmt)
        return list(res.scalars().all())

    async def query_submissions(
        self,
        *,
        class_id: Optional[str] = None,
        assignment_id: Optional[str] = None,
        student_id: Optional[str] = None,
        teacher_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> List[SubmissionModel]:
        """Query submissions with multi-tenancy filtering."""
        stmt = select(SubmissionModel)
        if class_id:
            stmt = stmt.where(SubmissionModel.class_id == class_id)
        if assignment_id:
            stmt = stmt.where(SubmissionModel.assignment_id == assignment_id)
        if student_id:
            stmt = stmt.where(SubmissionModel.student_id == student_id)
        if not is_admin and teacher_id and not class_id:
            stmt = stmt.where(
                (SubmissionModel.teacher_id == teacher_id) | (SubmissionModel.teacher_id.is_(None))
            )
        stmt = stmt.order_by(SubmissionModel.checked_at.desc())
        res = await self._session.execute(stmt)
        return list(res.scalars().all())

    async def create_submission(
        self,
        submission_data: Dict[str, Any],
        *,
        auto_commit: bool = True,
    ) -> SubmissionModel:
        """Create a submission record directly."""
        sub_obj, _ = await self.save_or_update(
            submission_data=submission_data,
            auto_commit=auto_commit,
        )
        return sub_obj

    async def save_or_update(
        self,
        *,
        submission_data: Dict[str, Any],
        auto_commit: bool = True,
    ) -> Tuple[SubmissionModel, bool]:
        """Idempotent upsert submission by client_submission_uuid."""
        client_uuid = submission_data.get("client_submission_uuid")
        sub_obj = None
        if client_uuid:
            sub_obj = await self.get_by_client_uuid(client_uuid)

        q_json = self._serialize_questions(
            submission_data.get("questions_results")
            or submission_data.get("questions_results_json")
            or []
        )
        checked_time = submission_data.get("checked_at") or datetime.now(timezone.utc)

        if sub_obj:
            sub_obj.assignment_id = submission_data.get("assignment_id", sub_obj.assignment_id)
            sub_obj.class_id = submission_data.get("class_id", sub_obj.class_id)
            sub_obj.student_id = submission_data.get("student_id", sub_obj.student_id)
            sub_obj.student_name = submission_data.get("student_name", sub_obj.student_name)
            sub_obj.variant = submission_data.get("variant", sub_obj.variant)
            sub_obj.checked_at = checked_time
            sub_obj.overall_score = submission_data.get("overall_score", sub_obj.overall_score)
            sub_obj.max_score = submission_data.get("max_score", sub_obj.max_score)
            sub_obj.final_grade = submission_data.get("final_grade", sub_obj.final_grade)
            sub_obj.teacher_reviewed_flags = submission_data.get(
                "teacher_reviewed_flags", sub_obj.teacher_reviewed_flags
            )
            sub_obj.questions_results_json = q_json
            if submission_data.get("teacher_id"):
                sub_obj.teacher_id = submission_data["teacher_id"]
            is_created = False
        else:
            sub_id = submission_data.get("submission_id") or f"sub_{uuid.uuid4().hex[:8]}"
            sub_obj = SubmissionModel(
                submission_id=sub_id,
                client_submission_uuid=client_uuid or str(uuid.uuid4()),
                assignment_id=submission_data["assignment_id"],
                class_id=submission_data["class_id"],
                student_id=submission_data["student_id"],
                student_name=submission_data.get("student_name", ""),
                variant=submission_data.get("variant", 1),
                checked_at=checked_time,
                overall_score=submission_data.get("overall_score", 0.0),
                max_score=submission_data.get("max_score", 8.0),
                final_grade=submission_data.get("final_grade", 2),
                teacher_reviewed_flags=submission_data.get("teacher_reviewed_flags", 0),
                questions_results_json=q_json,
                teacher_id=submission_data.get("teacher_id"),
                created_at=datetime.now(timezone.utc),
            )
            self._session.add(sub_obj)
            is_created = True

        await self._commit_or_flush(sub_obj, auto_commit=auto_commit)
        return sub_obj, is_created

    async def batch_create(
        self,
        submissions_data: List[Dict[str, Any]],
        *,
        auto_commit: bool = True,
    ) -> Tuple[int, int]:
        """Batch sync upsert submissions from ingestion payload (returns inserted, updated count)."""
        inserted = 0
        updated = 0
        for data in submissions_data:
            _, is_new = await self.save_or_update(submission_data=data, auto_commit=False)
            if is_new:
                inserted += 1
            else:
                updated += 1
        if auto_commit:
            await self._session.commit()
        else:
            await self._session.flush()
        return inserted, updated

    async def bulk_upsert(
        self,
        submissions_data: List[Dict[str, Any]],
        *,
        auto_commit: bool = True,
    ) -> Tuple[int, int]:
        """Alias for batch_create."""
        return await self.batch_create(submissions_data=submissions_data, auto_commit=auto_commit)

    async def clear_student_submissions(
        self,
        student_id: str,
        *,
        class_id: Optional[str] = None,
        auto_commit: bool = True,
    ) -> int:
        """Clear all submission history for a student."""
        stmt = delete(SubmissionModel).where(SubmissionModel.student_id == student_id)
        if class_id:
            stmt = stmt.where(SubmissionModel.class_id == class_id)
        res = await self._session.execute(stmt)
        if auto_commit:
            await self._session.commit()
        return res.rowcount or 0

    async def clear_class_submissions(
        self,
        class_id: str,
        *,
        auto_commit: bool = True,
    ) -> int:
        """Clear all submissions for a class."""
        stmt = delete(SubmissionModel).where(SubmissionModel.class_id == class_id)
        res = await self._session.execute(stmt)
        if auto_commit:
            await self._session.commit()
        return res.rowcount or 0

    async def clear_teacher_submissions(
        self,
        teacher_id: str,
        *,
        auto_commit: bool = True,
    ) -> int:
        """Clear all submissions associated with teacher (purge data)."""
        stmt = delete(SubmissionModel).where(SubmissionModel.teacher_id == teacher_id)
        res = await self._session.execute(stmt)
        if auto_commit:
            await self._session.commit()
        return res.rowcount or 0

    async def count_by_teacher(self, teacher_id: str) -> int:
        """Count submissions associated with teacher directly or via teacher's classes."""
        from app.models.school import ClassModel
        cls_subquery = select(ClassModel.class_id).where(ClassModel.teacher_uuid == teacher_id)
        stmt = select(func.count(SubmissionModel.submission_id)).where(
            (SubmissionModel.teacher_id == teacher_id) | (SubmissionModel.class_id.in_(cls_subquery))
        )
        res = await self._session.execute(stmt)
        return res.scalar() or 0

    def _serialize_questions(self, questions_data: Any) -> str:
        """Internal helper to serialize questions results into JSON string."""
        if isinstance(questions_data, str):
            return questions_data
        if hasattr(questions_data, "__iter__"):
            clean = []
            for item in questions_data:
                if hasattr(item, "model_dump"):
                    clean.append(item.model_dump())
                elif isinstance(item, dict):
                    clean.append(item)
            return json.dumps(clean, ensure_ascii=False)
        return "[]"
