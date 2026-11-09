from datetime import datetime, timezone
from typing import Optional, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, or_

from app.models.school import ClassModel, StudentModel, ClassAssignmentModel
from app.models.submission import SubmissionModel
from app.repositories.base import BaseRepository


class ClassRepository(BaseRepository[ClassModel]):
    """Data access repository for ClassModel."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, ClassModel)

    async def get_by_id(self, class_id: str) -> Optional[ClassModel]:
        """Fetch class by primary key class_id."""
        return await self._session.get(ClassModel, class_id)

    async def get_by_id_and_teacher(
        self,
        class_id: str,
        teacher_uuid: str,
    ) -> Optional[ClassModel]:
        """Fetch class and verify ownership by teacher_uuid (allows unassigned if legacy)."""
        stmt = select(ClassModel).where(
            ClassModel.class_id == class_id,
            or_(ClassModel.teacher_uuid == teacher_uuid, ClassModel.teacher_uuid.is_(None)),
        )
        res = await self._session.execute(stmt)
        return res.scalar_one_or_none()

    async def list_by_teacher(
        self,
        teacher_uuid: Optional[str] = None,
        *,
        include_unassigned: bool = True,
    ) -> List[ClassModel]:
        """List classes filtered by teacher_uuid."""
        stmt = select(ClassModel)
        if teacher_uuid:
            if include_unassigned:
                stmt = stmt.where(
                    or_(ClassModel.teacher_uuid == teacher_uuid, ClassModel.teacher_uuid.is_(None))
                )
            else:
                stmt = stmt.where(ClassModel.teacher_uuid == teacher_uuid)
        stmt = stmt.order_by(ClassModel.name)
        res = await self._session.execute(stmt)
        return list(res.scalars().all())

    async def count_by_teacher(self, teacher_uuid: str) -> int:
        """Count classes assigned to teacher."""
        stmt = select(func.count(ClassModel.class_id)).where(ClassModel.teacher_uuid == teacher_uuid)
        res = await self._session.execute(stmt)
        return res.scalar() or 0

    async def list_classes_with_student_counts(
        self,
        teacher_uuid: Optional[str] = None,
        *,
        is_admin: bool = False,
    ) -> List[Tuple[ClassModel, int]]:
        """List classes outer-joined with students to calculate active student counts."""
        stmt = (
            select(
                ClassModel,
                func.count(StudentModel.student_id).label("student_count"),
            )
            .outerjoin(
                StudentModel,
                (ClassModel.class_id == StudentModel.class_id) & (StudentModel.is_active != False),
            )
            .group_by(ClassModel.class_id)
        )
        if not is_admin and teacher_uuid:
            stmt = stmt.where(
                or_(ClassModel.teacher_uuid == teacher_uuid, ClassModel.teacher_uuid.is_(None))
            )

        res = await self._session.execute(stmt)
        return [(row[0], row[1]) for row in res.all()]

    async def create_class(
        self,
        *,
        class_id: str,
        name: str,
        teacher_uuid: Optional[str] = None,
        subject: str = "tatar_language",
        academic_year: str = "2026-2027",
        auto_commit: bool = True,
    ) -> ClassModel:
        """Create a new class record."""
        cls_obj = ClassModel(
            class_id=class_id,
            name=name,
            teacher_uuid=teacher_uuid,
            subject=subject,
            academic_year=academic_year,
            created_at=datetime.now(timezone.utc),
        )
        self._session.add(cls_obj)
        await self._commit_or_flush(cls_obj, auto_commit=auto_commit)
        return cls_obj

    async def get_or_create_class(
        self,
        class_id: str,
        *,
        name: Optional[str] = None,
        teacher_uuid: Optional[str] = None,
        auto_commit: bool = True,
    ) -> ClassModel:
        """Idempotent get or create for classes."""
        cls_obj = await self.get_by_id(class_id)
        if not cls_obj:
            cls_obj = await self.create_class(
                class_id=class_id,
                name=name or class_id,
                teacher_uuid=teacher_uuid,
                auto_commit=auto_commit,
            )
        return cls_obj

    async def delete_cascade(self, class_id: str, *, auto_commit: bool = True) -> bool:
        """Cascade delete class, its students, assignments, and submissions."""
        cls_obj = await self.get_by_id(class_id)
        if not cls_obj:
            return False

        await self._session.execute(
            delete(SubmissionModel).where(SubmissionModel.class_id == class_id)
        )
        await self._session.execute(
            delete(ClassAssignmentModel).where(ClassAssignmentModel.class_id == class_id)
        )
        await self._session.execute(
            delete(StudentModel).where(StudentModel.class_id == class_id)
        )
        await self._session.delete(cls_obj)

        if auto_commit:
            await self._session.commit()
        return True

    async def delete_class_cascade(self, class_id: str, *, auto_commit: bool = True) -> bool:
        """Alias for delete_cascade."""
        return await self.delete_cascade(class_id, auto_commit=auto_commit)

    async def delete_all_by_teacher(self, teacher_uuid: str, *, auto_commit: bool = True) -> int:
        """Delete all classes and associated records for a teacher (teacher data purge)."""
        classes = await self.list_by_teacher(teacher_uuid, include_unassigned=False)
        count = 0
        for cls_obj in classes:
            await self.delete_cascade(cls_obj.class_id, auto_commit=False)
            count += 1
        if auto_commit:
            await self._session.commit()
        return count
