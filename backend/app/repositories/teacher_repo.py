import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from app.models.teacher import TeacherModel
from app.models.user import UserModel
from app.repositories.base import BaseRepository


class TeacherRepository(BaseRepository[TeacherModel]):
    """Data access repository for TeacherModel."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, TeacherModel)

    async def get_by_uuid(self, teacher_uuid: str) -> Optional[TeacherModel]:
        """Fetch teacher profile by teacher_uuid."""
        stmt = select(TeacherModel).where(TeacherModel.teacher_uuid == teacher_uuid)
        res = await self._session.execute(stmt)
        return res.scalar_one_or_none()

    async def get_by_user_id(self, user_id: str) -> Optional[TeacherModel]:
        """Fetch teacher profile by linked user_id or teacher_uuid."""
        stmt = select(TeacherModel).where(
            or_(TeacherModel.user_id == user_id, TeacherModel.teacher_uuid == user_id)
        )
        res = await self._session.execute(stmt)
        return res.scalar_one_or_none()

    async def create_teacher(
        self,
        *,
        teacher_name: str,
        school_name: str,
        teacher_uuid: Optional[str] = None,
        user_id: Optional[str] = None,
        device_os: str = "android",
        app_version: str = "1.0.4",
        grade_5_min_pct: int = 85,
        grade_4_min_pct: int = 70,
        grade_3_min_pct: int = 50,
        confidence_flag_threshold: float = 0.65,
        auto_commit: bool = True,
    ) -> TeacherModel:
        """Create new teacher profile."""
        now = datetime.now(timezone.utc)
        teacher = TeacherModel(
            teacher_uuid=teacher_uuid or str(uuid.uuid4()),
            user_id=user_id,
            device_os=device_os,
            app_version=app_version,
            teacher_name=teacher_name.strip(),
            school_name=school_name.strip(),
            grade_5_min_pct=grade_5_min_pct,
            grade_4_min_pct=grade_4_min_pct,
            grade_3_min_pct=grade_3_min_pct,
            confidence_flag_threshold=confidence_flag_threshold,
            created_at=now,
            updated_at=now,
        )
        self._session.add(teacher)
        await self._commit_or_flush(teacher, auto_commit=auto_commit)
        return teacher

    async def update_teacher(
        self,
        teacher: TeacherModel,
        update_data: Dict[str, Any],
        *,
        auto_commit: bool = True,
    ) -> TeacherModel:
        """Update teacher profile attributes."""
        for key, value in update_data.items():
            if hasattr(teacher, key):
                setattr(teacher, key, value)
        teacher.updated_at = datetime.now(timezone.utc)
        await self._commit_or_flush(teacher, auto_commit=auto_commit)
        return teacher

    async def upsert_handshake(
        self,
        *,
        teacher_uuid: str,
        device_os: str,
        app_version: str,
        teacher_name: str,
        school_name: str,
        grading_scale: Dict[str, int],
        confidence_threshold: float = 0.65,
        auto_commit: bool = True,
    ) -> TeacherModel:
        """Upsert teacher settings during mobile device handshake."""
        teacher = await self.get_by_uuid(teacher_uuid)
        now = datetime.now(timezone.utc)
        g5 = grading_scale.get("grade_5_min_pct", 85)
        g4 = grading_scale.get("grade_4_min_pct", 70)
        g3 = grading_scale.get("grade_3_min_pct", 50)

        if not teacher:
            teacher = TeacherModel(
                teacher_uuid=teacher_uuid,
                device_os=device_os,
                app_version=app_version,
                teacher_name=teacher_name,
                school_name=school_name,
                grade_5_min_pct=g5,
                grade_4_min_pct=g4,
                grade_3_min_pct=g3,
                confidence_flag_threshold=confidence_threshold,
                created_at=now,
                updated_at=now,
            )
            self._session.add(teacher)
        else:
            teacher.device_os = device_os
            teacher.app_version = app_version
            teacher.teacher_name = teacher_name
            teacher.school_name = school_name
            teacher.grade_5_min_pct = g5
            teacher.grade_4_min_pct = g4
            teacher.grade_3_min_pct = g3
            teacher.confidence_flag_threshold = confidence_threshold
            teacher.updated_at = now

        await self._commit_or_flush(teacher, auto_commit=auto_commit)
        return teacher

    async def update_profile_settings(
        self,
        teacher_uuid_or_user_id: str,
        *,
        teacher_name: Optional[str] = None,
        school_name: Optional[str] = None,
        grading_scale: Optional[Dict[str, int]] = None,
        confidence_threshold: Optional[float] = None,
        auto_commit: bool = True,
    ) -> Optional[TeacherModel]:
        """Update teacher profile preferences."""
        teacher = await self.get_by_user_id(teacher_uuid_or_user_id)
        if not teacher:
            return None

        if teacher_name is not None:
            teacher.teacher_name = teacher_name.strip()
        if school_name is not None:
            teacher.school_name = school_name.strip()
        if grading_scale:
            if "grade_5_min_pct" in grading_scale:
                teacher.grade_5_min_pct = grading_scale["grade_5_min_pct"]
            if "grade_4_min_pct" in grading_scale:
                teacher.grade_4_min_pct = grading_scale["grade_4_min_pct"]
            if "grade_3_min_pct" in grading_scale:
                teacher.grade_3_min_pct = grading_scale["grade_3_min_pct"]
        if confidence_threshold is not None:
            teacher.confidence_flag_threshold = confidence_threshold

        teacher.updated_at = datetime.now(timezone.utc)
        await self._commit_or_flush(teacher, auto_commit=auto_commit)
        return teacher

    async def create_profile_for_user(
        self,
        user: UserModel,
        *,
        auto_commit: bool = True,
    ) -> TeacherModel:
        """Create default teacher profile linked to a UserModel."""
        now = datetime.now(timezone.utc)
        teacher = TeacherModel(
            user_id=user.id,
            teacher_uuid=user.id,
            teacher_name=user.full_name,
            school_name=user.school_name or "Школа",
            grade_5_min_pct=85,
            grade_4_min_pct=70,
            grade_3_min_pct=50,
            confidence_flag_threshold=0.65,
            created_at=now,
            updated_at=now,
        )
        self._session.add(teacher)
        await self._commit_or_flush(teacher, auto_commit=auto_commit)
        return teacher
