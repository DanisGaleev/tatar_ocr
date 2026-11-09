import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete

from app.models.school import ClassModel, ClassAssignmentModel, StudentModel
from app.models.test import AssembledTestModel
from app.models.submission import SubmissionModel
from app.repositories.base import BaseRepository


class AssignmentRepository(BaseRepository[ClassAssignmentModel]):
    """Data access repository for ClassAssignmentModel and AssembledTestModel."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, ClassAssignmentModel)

    async def get_by_id(self, assignment_id: str) -> Optional[AssembledTestModel]:
        """Fetch test definition by assignment_id / test_id."""
        return await self._session.get(AssembledTestModel, assignment_id)

    async def get_assembled_test(self, test_id: str) -> Optional[AssembledTestModel]:
        """Fetch test definition by test_id (alias for get_by_id)."""
        return await self._session.get(AssembledTestModel, test_id)

    async def list_all_assembled_tests(self) -> List[AssembledTestModel]:
        """List all assembled test definitions ordered by creation time descending."""
        stmt = select(AssembledTestModel).order_by(AssembledTestModel.created_at.desc())
        res = await self._session.execute(stmt)
        return list(res.scalars().all())

    async def get_class_assignment(
        self,
        class_id: str,
        assignment_id: str,
    ) -> Optional[ClassAssignmentModel]:
        """Fetch specific assignment link for a class."""
        stmt = select(ClassAssignmentModel).where(
            ClassAssignmentModel.class_id == class_id,
            ClassAssignmentModel.assignment_id == assignment_id,
        )
        res = await self._session.execute(stmt)
        return res.scalar_one_or_none()

    async def list_by_class(self, class_id: str) -> List[ClassAssignmentModel]:
        """List assignments assigned to a class."""
        stmt = (
            select(ClassAssignmentModel)
            .where(ClassAssignmentModel.class_id == class_id)
            .order_by(ClassAssignmentModel.assigned_at.desc())
        )
        res = await self._session.execute(stmt)
        return list(res.scalars().all())

    async def create_assignment(
        self,
        *,
        class_id: str,
        assignment_id: str,
        due_date: Optional[Union[str, datetime]] = None,
        status: str = "active",
        auto_commit: bool = True,
    ) -> ClassAssignmentModel:
        """Assign test to class or update existing assignment settings."""
        ca_obj = await self.get_class_assignment(class_id, assignment_id)
        due_dt = self._parse_due_date(due_date)

        if not ca_obj:
            ca_obj = ClassAssignmentModel(
                id=f"ca_{uuid.uuid4().hex[:8]}",
                class_id=class_id,
                assignment_id=assignment_id,
                due_date=due_dt,
                status=status,
                assigned_at=datetime.now(timezone.utc),
            )
            self._session.add(ca_obj)
        else:
            if due_dt is not None:
                ca_obj.due_date = due_dt
            if status:
                ca_obj.status = status

        await self._commit_or_flush(ca_obj, auto_commit=auto_commit)
        return ca_obj

    async def assign_test_to_class(
        self,
        *,
        class_id: str,
        assignment_id: str,
        due_date: Optional[Union[str, datetime]] = None,
        status: str = "active",
        auto_commit: bool = True,
    ) -> ClassAssignmentModel:
        """Alias for create_assignment."""
        return await self.create_assignment(
            class_id=class_id,
            assignment_id=assignment_id,
            due_date=due_date,
            status=status,
            auto_commit=auto_commit,
        )

    async def delete_assignment(
        self,
        class_id: str,
        assignment_id: str,
        *,
        auto_commit: bool = True,
    ) -> bool:
        """Delete assignment link for a class."""
        ca_obj = await self.get_class_assignment(class_id, assignment_id)
        if ca_obj:
            await self._session.delete(ca_obj)
            if auto_commit:
                await self._session.commit()
            return True
        return False

    async def delete_class_assignment(
        self,
        class_id: str,
        assignment_id: str,
        *,
        auto_commit: bool = True,
    ) -> bool:
        """Alias for delete_assignment."""
        return await self.delete_assignment(class_id, assignment_id, auto_commit=auto_commit)

    async def list_assignments_with_stats(
        self,
        *,
        class_id: Optional[str] = None,
        status_filter: Optional[str] = None,
        teacher_uuid: Optional[str] = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """List assignments aggregated with class details, student counts, and completion stats."""
        stmt = select(AssembledTestModel).order_by(AssembledTestModel.created_at.desc())
        res = await self._session.execute(stmt)
        tests = res.scalars().all()

        results = []
        for t in tests:
            ca_stmt = select(ClassAssignmentModel, ClassModel).join(
                ClassModel, ClassAssignmentModel.class_id == ClassModel.class_id
            ).where(ClassAssignmentModel.assignment_id == t.test_id)

            if class_id:
                ca_stmt = ca_stmt.where(ClassAssignmentModel.class_id == class_id)
            if status_filter:
                ca_stmt = ca_stmt.where(ClassAssignmentModel.status == status_filter)

            ca_res = await self._session.execute(ca_stmt)
            ca_rows = ca_res.all()
            if (class_id or status_filter) and not ca_rows:
                continue

            assigned_classes_list = []
            for ca_obj, cls_obj in ca_rows:
                if not is_admin and teacher_uuid:
                    if cls_obj.teacher_uuid and cls_obj.teacher_uuid != teacher_uuid:
                        continue

                stu_stmt = select(func.count(StudentModel.student_id)).where(
                    StudentModel.class_id == cls_obj.class_id,
                    StudentModel.is_active != False,
                )
                stu_res = await self._session.execute(stu_stmt)
                total_students = stu_res.scalar() or 0

                sub_stmt = select(func.count(SubmissionModel.submission_id)).where(
                    SubmissionModel.class_id == cls_obj.class_id,
                    SubmissionModel.assignment_id == t.test_id,
                )
                sub_res = await self._session.execute(sub_stmt)
                checked_count = sub_res.scalar() or 0

                assigned_classes_list.append({
                    "class_id": cls_obj.class_id,
                    "class_name": cls_obj.name,
                    "assigned_at": ca_obj.assigned_at.isoformat() if ca_obj.assigned_at else None,
                    "due_date": ca_obj.due_date.isoformat() if ca_obj.due_date else None,
                    "status": ca_obj.status,
                    "total_students": total_students,
                    "checked_submissions_count": checked_count,
                })

            results.append({
                "assignment_id": t.test_id,
                "title": t.title,
                "grade_level": t.grade_level,
                "total_variants": t.total_variants,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "assigned_classes": assigned_classes_list,
            })

        return results

    async def delete_all_by_teacher(self, teacher_uuid: str, *, auto_commit: bool = True) -> int:
        """Delete all class assignments belonging to a teacher's classes."""
        cls_subquery = select(ClassModel.class_id).where(ClassModel.teacher_uuid == teacher_uuid)
        stmt = delete(ClassAssignmentModel).where(ClassAssignmentModel.class_id.in_(cls_subquery))
        res = await self._session.execute(stmt)
        if auto_commit:
            await self._session.commit()
        return res.rowcount or 0

    def _parse_due_date(self, due_date: Optional[Union[str, datetime]]) -> Optional[datetime]:
        """Internal helper to parse due date."""
        if not due_date:
            return None
        if isinstance(due_date, datetime):
            return due_date
        try:
            return datetime.fromisoformat(due_date.replace("Z", "+00:00"))
        except Exception:
            return None
