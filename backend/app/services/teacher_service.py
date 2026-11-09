from datetime import datetime, timezone
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BusinessValidationError
from app.core.security import verify_password, get_password_hash, verify_totp_code
from app.models.user import UserModel
from app.repositories.teacher_repo import TeacherRepository
from app.repositories.user_repo import UserRepository
from app.repositories.verification_repo import VerificationCodeRepository
from app.repositories.session_repo import SessionRepository
from app.repositories.audit_repo import AuditRepository
from app.schemas.auth import GradingScale, TeacherPreferences
from app.schemas.user import (
    UserResponse,
    TeacherProfileResponse,
    TeacherProfileUpdateRequest,
    TeacherChangePasswordRequest,
)


class TeacherService:
    """Service layer managing teacher profile preferences, grading thresholds, and password change with 2FA."""

    def __init__(self, db: AsyncSession):
        self._db: AsyncSession = db
        self._teacher_repo = TeacherRepository(db)
        self._user_repo = UserRepository(db)
        self._verification_repo = VerificationCodeRepository(db)
        self._session_repo = SessionRepository(db)
        self._audit_repo = AuditRepository(db)

    async def get_profile(self, current_user: UserModel) -> TeacherProfileResponse:
        teacher = await self._teacher_repo.get_by_user_id(current_user.id)
        if not teacher:
            teacher = await self._teacher_repo.create_profile_for_user(current_user)

        return TeacherProfileResponse(
            user=UserResponse.model_validate(current_user),
            preferences=TeacherPreferences(
                grading_scale=GradingScale(
                    grade_5_min_pct=teacher.grade_5_min_pct,
                    grade_4_min_pct=teacher.grade_4_min_pct,
                    grade_3_min_pct=teacher.grade_3_min_pct,
                ),
                confidence_flag_threshold=teacher.confidence_flag_threshold,
            ),
        )

    async def update_profile(
        self,
        current_user: UserModel,
        payload: TeacherProfileUpdateRequest,
    ) -> TeacherProfileResponse:
        user_updates: Dict[str, Any] = {}
        if payload.full_name is not None:
            user_updates["full_name"] = payload.full_name.strip()
        if payload.school_name is not None:
            user_updates["school_name"] = payload.school_name.strip()

        if user_updates:
            current_user = await self._user_repo.update_user(current_user, user_updates)

        teacher = await self._teacher_repo.get_by_user_id(current_user.id)
        if not teacher:
            teacher = await self._teacher_repo.create_profile_for_user(current_user)

        teacher_updates: Dict[str, Any] = {
            "teacher_name": current_user.full_name,
            "school_name": current_user.school_name or teacher.school_name,
        }
        if payload.grading_scale:
            teacher_updates["grade_5_min_pct"] = payload.grading_scale.grade_5_min_pct
            teacher_updates["grade_4_min_pct"] = payload.grading_scale.grade_4_min_pct
            teacher_updates["grade_3_min_pct"] = payload.grading_scale.grade_3_min_pct
        if payload.confidence_flag_threshold is not None:
            teacher_updates["confidence_flag_threshold"] = float(payload.confidence_flag_threshold)

        teacher = await self._teacher_repo.update_teacher(teacher, teacher_updates)

        return TeacherProfileResponse(
            user=UserResponse.model_validate(current_user),
            preferences=TeacherPreferences(
                grading_scale=GradingScale(
                    grade_5_min_pct=teacher.grade_5_min_pct,
                    grade_4_min_pct=teacher.grade_4_min_pct,
                    grade_3_min_pct=teacher.grade_3_min_pct,
                ),
                confidence_flag_threshold=teacher.confidence_flag_threshold,
            ),
        )

    async def change_password(
        self,
        current_user: UserModel,
        payload: TeacherChangePasswordRequest,
    ) -> Dict[str, str]:
        if not current_user.hashed_password or not verify_password(payload.old_password, current_user.hashed_password):
            raise BusinessValidationError(message="Invalid old password")

        if len(payload.new_password) < 8:
            raise BusinessValidationError(message="New password must be at least 8 characters long")

        # 2FA Verification if active
        if current_user.two_factor_enabled:
            if not payload.two_factor_code:
                raise BusinessValidationError(message="Two-factor verification code required")
            if current_user.two_factor_method == "totp":
                if not current_user.totp_secret or not verify_totp_code(current_user.totp_secret, payload.two_factor_code):
                    raise BusinessValidationError(message="Invalid 2FA code")
            elif current_user.two_factor_method == "email":
                vcd = await self._verification_repo.get_valid_code(current_user.email, "login_2fa")
                if not vcd:
                    vcd = await self._verification_repo.get_valid_code(current_user.email, "password_reset")
                if not vcd or vcd.attempts_left <= 0:
                    raise BusinessValidationError(message="Invalid 2FA code")
                if vcd.code != payload.two_factor_code.strip():
                    await self._verification_repo.decrement_attempts(vcd)
                    raise BusinessValidationError(message="Invalid 2FA code")
                await self._verification_repo.consume_code(vcd)

        await self._user_repo.update_password(current_user, get_password_hash(payload.new_password))
        await self._session_repo.revoke_all_for_user(current_user.id)

        await self._audit_repo.log_event(
            action="CHANGE_PASSWORD",
            user_id=current_user.id,
            user_email=current_user.email,
            role=current_user.role,
            target_entity="user",
            target_id=current_user.id,
        )

        return {
            "status": "success",
            "message": "Password changed successfully. Please log in again.",
        }
