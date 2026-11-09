import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete

from app.models.school import StudentModel, ClassModel
from app.repositories.base import BaseRepository


class StudentRepository(BaseRepository[StudentModel]):
    """Data access repository for StudentModel."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, StudentModel)

    async def get_by_id(
        self,
        student_id: str,
        class_id: Optional[str] = None,
    ) -> Optional[StudentModel]:
        """Fetch student by student_id and optional class_id."""
        stmt = select(StudentModel).where(StudentModel.student_id == student_id)
        if class_id:
            stmt = stmt.where(StudentModel.class_id == class_id)
        res = await self._session.execute(stmt)
        return res.scalar_one_or_none()

    async def list_by_class(
        self,
        class_id: str,
        *,
        active_only: bool = True,
    ) -> List[StudentModel]:
        """List students in class ordered by last_name, first_name."""
        stmt = select(StudentModel).where(StudentModel.class_id == class_id)
        if active_only:
            stmt = stmt.where(StudentModel.is_active != False)
        stmt = stmt.order_by(StudentModel.last_name, StudentModel.first_name)
        res = await self._session.execute(stmt)
        return list(res.scalars().all())

    async def create_student(
        self,
        *,
        class_id: str,
        last_name: str,
        first_name: str,
        middle_name: str = "",
        short_name: Optional[str] = None,
        student_id: Optional[str] = None,
        is_active: bool = True,
        auto_commit: bool = True,
    ) -> StudentModel:
        """Create a single student record with automated name normalization."""
        full_name, formatted_short = self._format_names(
            last_name=last_name,
            first_name=first_name,
            middle_name=middle_name,
            short_name=short_name,
        )
        student = StudentModel(
            student_id=student_id or f"stu_{uuid.uuid4().hex[:6]}",
            class_id=class_id,
            last_name=last_name.strip(),
            first_name=first_name.strip(),
            middle_name=middle_name.strip(),
            full_name=full_name,
            short_name=formatted_short,
            is_active=is_active,
            created_at=datetime.now(timezone.utc),
        )
        self._session.add(student)
        await self._commit_or_flush(student, auto_commit=auto_commit)
        return student

    async def bulk_create(
        self,
        students_data: List[Dict[str, Any]],
        *,
        auto_commit: bool = True,
    ) -> List[StudentModel]:
        """Bulk insert students from structured list."""
        created = []
        for item in students_data:
            full_name, short_name = self._format_names(
                last_name=item["last_name"],
                first_name=item.get("first_name", ""),
                middle_name=item.get("middle_name", ""),
                short_name=item.get("short_name"),
            )
            stu = StudentModel(
                student_id=item.get("student_id") or f"stu_{uuid.uuid4().hex[:6]}",
                class_id=item["class_id"],
                last_name=item["last_name"].strip(),
                first_name=item.get("first_name", "").strip(),
                middle_name=item.get("middle_name", "").strip(),
                full_name=full_name,
                short_name=short_name,
                is_active=item.get("is_active", True),
                created_at=datetime.now(timezone.utc),
            )
            self._session.add(stu)
            created.append(stu)
        if auto_commit:
            await self._session.commit()
            for s in created:
                await self._session.refresh(s)
        else:
            await self._session.flush()
        return created

    async def update_student(
        self,
        student: StudentModel,
        *,
        last_name: Optional[str] = None,
        first_name: Optional[str] = None,
        middle_name: Optional[str] = None,
        short_name: Optional[str] = None,
        auto_commit: bool = True,
    ) -> StudentModel:
        """Update student names and recalculate composite names."""
        ln = last_name.strip() if last_name is not None else student.last_name
        fn = first_name.strip() if first_name is not None else student.first_name
        mn = middle_name.strip() if middle_name is not None else (student.middle_name or "")
        full_name, formatted_short = self._format_names(
            last_name=ln,
            first_name=fn,
            middle_name=mn,
            short_name=short_name,
        )
        student.last_name = ln
        student.first_name = fn
        student.middle_name = mn
        student.full_name = full_name
        student.short_name = formatted_short
        await self._commit_or_flush(student, auto_commit=auto_commit)
        return student

    async def soft_delete(
        self,
        student_id: str,
        class_id: Optional[str] = None,
        *,
        auto_commit: bool = True,
    ) -> Optional[StudentModel]:
        """Soft delete student by setting is_active = False."""
        student = await self.get_by_id(student_id, class_id=class_id)
        if student:
            student.is_active = False
            await self._commit_or_flush(student, auto_commit=auto_commit)
            return student
        return None

    async def count_by_class(
        self,
        class_id: str,
        *,
        active_only: bool = True,
    ) -> int:
        """Count students in class."""
        stmt = select(func.count(StudentModel.student_id)).where(StudentModel.class_id == class_id)
        if active_only:
            stmt = stmt.where(StudentModel.is_active != False)
        res = await self._session.execute(stmt)
        return res.scalar() or 0

    async def count_by_teacher(
        self,
        teacher_uuid: str,
        *,
        active_only: bool = True,
    ) -> int:
        """Count students across all classes owned by teacher."""
        stmt = (
            select(func.count(StudentModel.student_id))
            .join(ClassModel, StudentModel.class_id == ClassModel.class_id)
            .where(ClassModel.teacher_uuid == teacher_uuid)
        )
        if active_only:
            stmt = stmt.where(StudentModel.is_active != False)
        res = await self._session.execute(stmt)
        return res.scalar() or 0

    async def clear_class_students(
        self,
        class_id: str,
        *,
        auto_commit: bool = True,
    ) -> int:
        """Hard delete all students in class."""
        stmt = delete(StudentModel).where(StudentModel.class_id == class_id)
        res = await self._session.execute(stmt)
        if auto_commit:
            await self._session.commit()
        return res.rowcount or 0

    async def delete_all_by_class(
        self,
        class_id: str,
        *,
        auto_commit: bool = True,
    ) -> int:
        """Alias for clear_class_students."""
        return await self.clear_class_students(class_id, auto_commit=auto_commit)

    def _format_names(
        self,
        *,
        last_name: str,
        first_name: str,
        middle_name: str = "",
        short_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Internal helper for full_name and short_name formatting."""
        ln = last_name.strip()
        fn = first_name.strip()
        mn = middle_name.strip()
        full_name = f"{ln} {fn} {mn}".strip() if mn else f"{ln} {fn}".strip()
        if short_name:
            calc_short = short_name.strip()
        elif fn:
            calc_short = f"{ln} {fn[0]}."
        else:
            calc_short = ln
        return full_name, calc_short
