from datetime import datetime, timezone
import json
import logging
import secrets
import string
from typing import Any, Dict, List, Optional
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    BusinessValidationError,
    NotFoundError,
    PermissionDeniedError,
)
from app.core.security import get_password_hash
from app.core.telemetry import telemetry_manager
from app.models.school import ClassAssignmentModel, ClassModel, StudentModel
from app.models.session import UserSessionModel
from app.models.submission import SubmissionModel
from app.models.user import UserModel
from app.repositories.assignment_repo import AssignmentRepository
from app.repositories.audit_repo import AuditRepository
from app.repositories.class_repo import ClassRepository
from app.repositories.session_repo import SessionRepository
from app.repositories.student_repo import StudentRepository
from app.repositories.submission_repo import SubmissionRepository
from app.repositories.teacher_repo import TeacherRepository
from app.repositories.user_repo import UserRepository
from app.schemas.admin import (
    AddStudentRequest,
    AdminActionResponse,
    AdminActiveSessionsResponse,
    AdminAuditLogItem,
    AdminAuditLogsResponse,
    AdminClearDataResponse,
    AdminClearStudentDataResponse,
    AdminDeleteStudentResponse,
    AdminFreezeResponse,
    AdminKillSessionResponse,
    AdminResetPasswordRequest,
    AdminResetPasswordResponse,
    AdminSessionItem,
    AdminUpdateProfileRequest,
    AdminUserDetailResponse,
    AdminUserItem,
    AdminUserListResponse,
    AdminUserStats,
)
from app.schemas.auth import TeacherPreferences
from app.schemas.classes import ClassItem, StudentRosterItem

logger = logging.getLogger(__name__)


class AdminService:
    """Business logic service for platform administration, user management, and auditing."""

    def __init__(self, db: AsyncSession):
        self._db: AsyncSession = db
        self._user_repo = UserRepository(db)
        self._session_repo = SessionRepository(db)
        self._audit_repo = AuditRepository(db)
        self._class_repo = ClassRepository(db)
        self._student_repo = StudentRepository(db)
        self._submission_repo = SubmissionRepository(db)
        self._assignment_repo = AssignmentRepository(db)
        self._teacher_repo = TeacherRepository(db)

    async def list_users(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        role: Optional[str] = None,
        search: Optional[str] = None,
    ) -> AdminUserListResponse:
        """Paginated list of platform users with multi-parameter filtering."""
        users, total = await self._user_repo.list_users(
            page=page,
            page_size=page_size,
            status=status,
            role=role,
            search=search,
        )
        items = [AdminUserItem.model_validate(u) for u in users]
        pages = (total + page_size - 1) // page_size if total > 0 else 1
        return AdminUserListResponse(
            total=total,
            page=page,
            page_size=page_size,
            pages=pages,
            users=items,
            items=items,
        )

    async def get_user_detail(self, user_id: str) -> AdminUserDetailResponse:
        """Full teacher card with aggregated statistics and recent sessions."""
        user = await self._user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundError(
                message=f"User '{user_id}' not found",
                details={"user_id": user_id},
            )

        # Classes and student breakdown
        classes = await self._class_repo.list_by_teacher(user_id, include_unassigned=False)
        class_ids = [c.class_id for c in classes]

        class_items: List[ClassItem] = []
        for c in classes:
            sc = await self._student_repo.count_by_class(c.class_id, active_only=True)
            class_items.append(
                ClassItem(
                    class_id=c.class_id,
                    name=c.name,
                    subject=c.subject,
                    academic_year=c.academic_year,
                    student_count=sc,
                )
            )

        # Aggregated stats
        students_count = await self._student_repo.count_by_teacher(user_id, active_only=True)
        submissions_count = await self._submission_repo.count_by_teacher(user_id)

        # Assignments count
        assignments_count = 0
        if class_ids:
            ca_stmt = select(func.count(ClassAssignmentModel.id)).where(
                ClassAssignmentModel.class_id.in_(class_ids)
            )
            ca_res = await self._db.execute(ca_stmt)
            assignments_count = ca_res.scalar() or 0

        # Average grade
        avg_grade = 0.0
        if class_ids:
            avg_stmt = select(func.avg(SubmissionModel.final_grade)).where(
                or_(SubmissionModel.teacher_id == user_id, SubmissionModel.class_id.in_(class_ids))
            )
        else:
            avg_stmt = select(func.avg(SubmissionModel.final_grade)).where(
                SubmissionModel.teacher_id == user_id
            )
        avg_res = await self._db.execute(avg_stmt)
        raw_avg = avg_res.scalar()
        if raw_avg is not None:
            avg_grade = round(float(raw_avg), 2)

        # Active sessions count
        now = datetime.now(timezone.utc)
        ses_cnt_stmt = select(func.count(UserSessionModel.id)).where(
            UserSessionModel.user_id == user_id,
            UserSessionModel.revoked_at.is_(None),
            UserSessionModel.expires_at > now,
        )
        ses_cnt_res = await self._db.execute(ses_cnt_stmt)
        active_sessions_count = ses_cnt_res.scalar() or 0

        stats = AdminUserStats(
            classes_count=len(classes),
            students_count=students_count,
            assignments_count=assignments_count,
            submissions_count=submissions_count,
            active_sessions_count=active_sessions_count,
            average_grade=avg_grade,
            last_login_at=user.last_login_at,
            last_active_at=user.last_active_at,
        )

        # Recent sessions
        sessions_db = await self._session_repo.list_by_user(user_id, active_only=False, limit=10)
        recent_sessions = [
            AdminSessionItem(
                id=s.id,
                user_id=s.user_id,
                user_email=user.email,
                user_full_name=user.full_name,
                device_info=s.device_info,
                ip_address=s.ip_address,
                user_agent=s.user_agent,
                created_at=s.created_at,
                expires_at=s.expires_at,
                revoked_at=s.revoked_at,
            )
            for s in sessions_db
        ]

        # Teacher preferences if present
        preferences = None
        teacher_rec = await self._teacher_repo.get_by_uuid(user_id)
        if not teacher_rec:
            teacher_rec = await self._teacher_repo.get_by_user_id(user_id)
        if teacher_rec:
            from app.schemas.auth import GradingScale
            preferences = TeacherPreferences(
                teacher_name=teacher_rec.teacher_name,
                school_name=teacher_rec.school_name,
                grading_scale=GradingScale(
                    grade_5_min_pct=teacher_rec.grade_5_min_pct,
                    grade_4_min_pct=teacher_rec.grade_4_min_pct,
                    grade_3_min_pct=teacher_rec.grade_3_min_pct,
                ),
                confidence_flag_threshold=teacher_rec.confidence_flag_threshold,
            )

        return AdminUserDetailResponse(
            user=AdminUserItem.model_validate(user),
            stats=stats,
            recent_sessions=recent_sessions,
            classes=class_items,
            preferences=preferences,
        )

    async def freeze_user(
        self,
        user_id: str,
        current_admin: UserModel,
        reason: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AdminFreezeResponse:
        """Freeze user account and revoke all active sessions immediately."""
        if user_id == current_admin.id:
            raise BusinessValidationError(message="Cannot freeze your own account")

        target_user = await self._user_repo.get_by_id(user_id)
        if not target_user:
            raise NotFoundError(
                message=f"User '{user_id}' not found",
                details={"user_id": user_id},
            )

        if target_user.status == "frozen":
            raise BusinessValidationError(message="User is already frozen")

        await self._user_repo.set_status(target_user, "frozen")
        sessions_revoked = await self._session_repo.revoke_all_for_user(user_id)

        # Audit log
        await self._audit_repo.log_event(
            action="FREEZE_USER",
            user_id=current_admin.id,
            user_email=current_admin.email,
            role=current_admin.role,
            target_entity="user",
            target_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={
                "admin_id": current_admin.id,
                "reason": reason,
                "sessions_revoked": sessions_revoked,
                "target_email": target_user.email,
            },
        )

        # Live telemetry broadcast
        await telemetry_manager.broadcast({
            "action": "FREEZE_USER",
            "user_id": user_id,
            "user_name": target_user.full_name,
            "school": target_user.school_name,
            "ip_address": ip_address,
            "details": {
                "admin_id": current_admin.id,
                "reason": reason,
                "sessions_revoked": sessions_revoked,
            },
        })

        return AdminFreezeResponse(
            status="frozen",
            user_id=user_id,
            sessions_revoked=sessions_revoked,
            message="User frozen and all active sessions revoked",
            reason=reason,
        )

    async def unfreeze_user(
        self,
        user_id: str,
        current_admin: UserModel,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AdminActionResponse:
        """Unfreeze user account, returning status to active and clearing lockout."""
        target_user = await self._user_repo.get_by_id(user_id)
        if not target_user:
            raise NotFoundError(
                message=f"User '{user_id}' not found",
                details={"user_id": user_id},
            )

        if target_user.status != "frozen":
            raise BusinessValidationError(message="User is not frozen")

        await self._user_repo.set_status(target_user, "active")
        await self._user_repo.reset_failed_attempts(target_user)

        # Audit log
        await self._audit_repo.log_event(
            action="UNFREEZE_USER",
            user_id=current_admin.id,
            user_email=current_admin.email,
            role=current_admin.role,
            target_entity="user",
            target_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={"admin_id": current_admin.id, "target_email": target_user.email},
        )

        # Live telemetry broadcast
        await telemetry_manager.broadcast({
            "action": "UNFREEZE_USER",
            "user_id": user_id,
            "user_name": target_user.full_name,
            "school": target_user.school_name,
            "ip_address": ip_address,
            "details": {"admin_id": current_admin.id},
        })

        return AdminActionResponse(
            status="active",
            user_id=user_id,
            message="User successfully unfrozen",
            details=None,
        )

    async def delete_user(
        self,
        user_id: str,
        current_admin: UserModel,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AdminActionResponse:
        """Soft delete user account, anonymize personal details, and revoke sessions."""
        if user_id == current_admin.id:
            raise BusinessValidationError(message="Cannot delete your own admin account")

        target_user = await self._user_repo.get_by_id(user_id)
        if not target_user:
            raise NotFoundError(
                message=f"User '{user_id}' not found",
                details={"user_id": user_id},
            )

        original_email = target_user.email
        await self._user_repo.soft_delete(target_user)
        target_user.full_name = "Удаленный пользователь"
        target_user.email = f"deleted_{user_id}@anonymized.local"
        target_user.school_name = None
        await self._db.commit()

        sessions_revoked = await self._session_repo.revoke_all_for_user(user_id)

        # Audit log
        await self._audit_repo.log_event(
            action="DELETE_USER",
            user_id=current_admin.id,
            user_email=current_admin.email,
            role=current_admin.role,
            target_entity="user",
            target_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={"admin_id": current_admin.id, "target_email": original_email, "sessions_revoked": sessions_revoked},
        )

        # Live telemetry broadcast
        await telemetry_manager.broadcast({
            "action": "DELETE_USER",
            "user_id": user_id,
            "user_name": "Удаленный пользователь",
            "ip_address": ip_address,
            "details": {"admin_id": current_admin.id, "sessions_revoked": sessions_revoked},
        })

        return AdminActionResponse(
            status="deleted",
            user_id=user_id,
            message="User successfully deleted and sessions revoked",
            details={"sessions_revoked": sessions_revoked},
        )

    async def reset_password(
        self,
        user_id: str,
        payload: AdminResetPasswordRequest,
        current_admin: UserModel,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AdminResetPasswordResponse:
        """Reset user password, auto-generating strong password if omitted, and revoke sessions."""
        target_user = await self._user_repo.get_by_id(user_id)
        if not target_user:
            raise NotFoundError(
                message=f"User '{user_id}' not found",
                details={"user_id": user_id},
            )

        if payload.new_password is not None:
            if len(payload.new_password) < 8:
                raise BusinessValidationError(message="Password must be at least 8 characters long")
            chosen_password = payload.new_password
        else:
            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
            chosen_password = "".join(secrets.choice(alphabet) for _ in range(12))

        await self._user_repo.update_password(
            target_user,
            get_password_hash(chosen_password),
        )
        sessions_revoked = await self._session_repo.revoke_all_for_user(user_id)

        # Audit log
        await self._audit_repo.log_event(
            action="ADMIN_RESET_PASSWORD",
            user_id=current_admin.id,
            user_email=current_admin.email,
            role=current_admin.role,
            target_entity="user",
            target_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={
                "admin_id": current_admin.id,
                "target_email": target_user.email,
                "email_notified": payload.send_email,
                "sessions_revoked": sessions_revoked,
            },
        )

        # Live telemetry broadcast
        await telemetry_manager.broadcast({
            "action": "ADMIN_RESET_PASSWORD",
            "user_id": user_id,
            "user_name": target_user.full_name,
            "school": target_user.school_name,
            "ip_address": ip_address,
            "details": {"admin_id": current_admin.id, "email_notified": payload.send_email},
        })

        return AdminResetPasswordResponse(
            status="password_reset",
            user_id=user_id,
            temporary_password=chosen_password,
            new_password=chosen_password,
            message="Password successfully reset and sessions revoked",
        )

    async def update_profile(
        self,
        user_id: str,
        payload: AdminUpdateProfileRequest,
        current_admin: UserModel,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AdminUserItem:
        """Update user profile by administrator and keep teacher profile synchronized."""
        target_user = await self._user_repo.get_by_id(user_id)
        if not target_user:
            raise NotFoundError(
                message=f"User '{user_id}' not found",
                details={"user_id": user_id},
            )

        if payload.email and payload.email.lower().strip() != target_user.email.lower():
            clean_email = payload.email.lower().strip()
            existing = await self._user_repo.get_by_email(clean_email)
            if existing and existing.id != user_id:
                raise BusinessValidationError(message="Email is already registered")

        updates: Dict[str, Any] = {}
        if payload.full_name is not None:
            updates["full_name"] = payload.full_name.strip()
        if payload.school_name is not None:
            updates["school_name"] = payload.school_name.strip() if payload.school_name else None
        if payload.email is not None:
            updates["email"] = payload.email.lower().strip()
        if payload.role is not None:
            updates["role"] = payload.role.strip()
        if payload.status is not None:
            updates["status"] = payload.status.strip()

        if updates:
            await self._user_repo.update_user(target_user, updates)

            # Sync TeacherModel if exists
            teacher_updates: Dict[str, Any] = {}
            if "full_name" in updates:
                teacher_updates["teacher_name"] = updates["full_name"]
            if "school_name" in updates:
                teacher_updates["school_name"] = updates["school_name"]
            if teacher_updates:
                await self._teacher_repo.update_profile_settings(
                    user_id,
                    **teacher_updates,
                )

        # Audit log
        await self._audit_repo.log_event(
            action="ADMIN_UPDATE_USER_PROFILE",
            user_id=current_admin.id,
            user_email=current_admin.email,
            role=current_admin.role,
            target_entity="user",
            target_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={"admin_id": current_admin.id, "updated_fields": list(updates.keys())},
        )

        # Live telemetry broadcast
        await telemetry_manager.broadcast({
            "action": "ADMIN_UPDATE_USER_PROFILE",
            "user_id": user_id,
            "user_name": target_user.full_name,
            "school": target_user.school_name,
            "ip_address": ip_address,
            "details": {"admin_id": current_admin.id, "updated_fields": list(updates.keys())},
        })

        return AdminUserItem.model_validate(target_user)

    async def clear_teacher_data(
        self,
        user_id: str,
        current_admin: UserModel,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AdminClearDataResponse:
        """Purge all educational data of teacher (classes, assignments, submissions) while retaining account."""
        target_user = await self._user_repo.get_by_id(user_id)
        if not target_user:
            raise NotFoundError(
                message=f"User '{user_id}' not found",
                details={"user_id": user_id},
            )

        classes = await self._class_repo.list_by_teacher(user_id, include_unassigned=False)
        class_ids = [c.class_id for c in classes]

        # 1. Delete submissions
        if class_ids:
            sub_stmt = delete(SubmissionModel).where(
                or_(SubmissionModel.teacher_id == user_id, SubmissionModel.class_id.in_(class_ids))
            )
        else:
            sub_stmt = delete(SubmissionModel).where(SubmissionModel.teacher_id == user_id)
        sub_res = await self._db.execute(sub_stmt)
        deleted_submissions = sub_res.rowcount or 0

        # 2. Delete class assignments
        deleted_assignments = 0
        if class_ids:
            ca_stmt = delete(ClassAssignmentModel).where(
                ClassAssignmentModel.class_id.in_(class_ids)
            )
            ca_res = await self._db.execute(ca_stmt)
            deleted_assignments = ca_res.rowcount or 0

        # 3. Delete students
        deleted_students = 0
        if class_ids:
            stu_stmt = delete(StudentModel).where(StudentModel.class_id.in_(class_ids))
            stu_res = await self._db.execute(stu_stmt)
            deleted_students = stu_res.rowcount or 0

        # 4. Delete classes
        deleted_classes = 0
        if class_ids:
            cls_stmt = delete(ClassModel).where(ClassModel.class_id.in_(class_ids))
            cls_res = await self._db.execute(cls_stmt)
            deleted_classes = cls_res.rowcount or 0

        await self._db.commit()

        # Audit log
        await self._audit_repo.log_event(
            action="CLEAR_USER_DATA",
            user_id=current_admin.id,
            user_email=current_admin.email,
            role=current_admin.role,
            target_entity="user",
            target_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={
                "admin_id": current_admin.id,
                "deleted_classes": deleted_classes,
                "deleted_assignments": deleted_assignments,
                "deleted_students": deleted_students,
                "deleted_submissions": deleted_submissions,
            },
        )

        # Live telemetry broadcast
        await telemetry_manager.broadcast({
            "action": "CLEAR_USER_DATA",
            "user_id": user_id,
            "user_name": target_user.full_name,
            "school": target_user.school_name,
            "ip_address": ip_address,
            "details": {
                "admin_id": current_admin.id,
                "deleted_classes": deleted_classes,
                "deleted_submissions": deleted_submissions,
            },
        })

        return AdminClearDataResponse(
            status="data_cleared",
            user_id=user_id,
            deleted_classes_count=deleted_classes,
            deleted_submissions_count=deleted_submissions,
            deleted_assignments_count=deleted_assignments,
            deleted_students_count=deleted_students,
            classes_cleared=deleted_classes,
            submissions_cleared=deleted_submissions,
            message="Teacher data cleared successfully",
        )

    async def add_student_for_teacher(
        self,
        *,
        user_id: str,
        class_id: str,
        payload: AddStudentRequest,
        current_admin: UserModel,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> StudentRosterItem:
        """Add a student to a specific class belonging to teacher."""
        target_user = await self._user_repo.get_by_id(user_id)
        if not target_user:
            raise NotFoundError(
                message=f"User '{user_id}' not found",
                details={"user_id": user_id},
            )

        cls_obj = await self._class_repo.get_by_id(class_id)
        if not cls_obj or (cls_obj.teacher_uuid and cls_obj.teacher_uuid != user_id):
            raise NotFoundError(
                message=f"Class '{class_id}' not found for teacher '{user_id}'.",
                details={"class_id": class_id, "teacher_id": user_id},
            )

        if not payload.last_name.strip() or not payload.first_name.strip():
            raise BusinessValidationError(message="Last name and first name are required")

        student = await self._student_repo.create_student(
            class_id=class_id,
            last_name=payload.last_name.strip(),
            first_name=payload.first_name.strip(),
            middle_name=(payload.middle_name or "").strip(),
            short_name=payload.short_name.strip() if payload.short_name else None,
            student_id=payload.student_id,
        )

        # Audit log
        await self._audit_repo.log_event(
            action="ADMIN_ADD_STUDENT",
            user_id=current_admin.id,
            user_email=current_admin.email,
            role=current_admin.role,
            target_entity="student",
            target_id=student.student_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={
                "admin_id": current_admin.id,
                "teacher_id": user_id,
                "class_id": class_id,
                "student_name": student.full_name,
            },
        )

        # Live telemetry broadcast
        await telemetry_manager.broadcast({
            "action": "ADMIN_ADD_STUDENT",
            "user_id": user_id,
            "user_name": target_user.full_name,
            "school": target_user.school_name,
            "ip_address": ip_address,
            "details": {
                "admin_id": current_admin.id,
                "teacher_id": user_id,
                "class_id": class_id,
                "student_id": student.student_id,
                "student_name": student.full_name,
            },
        })

        return StudentRosterItem(
            student_id=student.student_id,
            last_name=student.last_name,
            first_name=student.first_name,
            middle_name=student.middle_name or "",
            full_name=student.full_name,
            short_name=student.short_name,
        )

    async def delete_student_for_teacher(
        self,
        *,
        user_id: str,
        class_id: str,
        student_id: str,
        current_admin: UserModel,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AdminDeleteStudentResponse:
        """Soft delete a student from teacher's class."""
        target_user = await self._user_repo.get_by_id(user_id)
        if not target_user:
            raise NotFoundError(
                message=f"User '{user_id}' not found",
                details={"user_id": user_id},
            )

        cls_obj = await self._class_repo.get_by_id(class_id)
        if not cls_obj or (cls_obj.teacher_uuid and cls_obj.teacher_uuid != user_id):
            raise NotFoundError(
                message=f"Class '{class_id}' not found for teacher '{user_id}'.",
                details={"class_id": class_id, "teacher_id": user_id},
            )

        student = await self._student_repo.get_by_id(student_id, class_id=class_id)
        if not student:
            raise NotFoundError(
                message=f"Student '{student_id}' not found in class '{class_id}'.",
                details={"student_id": student_id, "class_id": class_id},
            )

        await self._student_repo.soft_delete(student_id, class_id=class_id)

        # Audit log
        await self._audit_repo.log_event(
            action="ADMIN_DELETE_STUDENT",
            user_id=current_admin.id,
            user_email=current_admin.email,
            role=current_admin.role,
            target_entity="student",
            target_id=student_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={
                "admin_id": current_admin.id,
                "teacher_id": user_id,
                "class_id": class_id,
                "student_id": student_id,
            },
        )

        # Live telemetry broadcast
        await telemetry_manager.broadcast({
            "action": "ADMIN_DELETE_STUDENT",
            "user_id": user_id,
            "user_name": target_user.full_name,
            "school": target_user.school_name,
            "ip_address": ip_address,
            "details": {
                "admin_id": current_admin.id,
                "teacher_id": user_id,
                "class_id": class_id,
                "student_id": student_id,
            },
        })

        return AdminDeleteStudentResponse(
            status="deleted",
            student_id=student_id,
            class_id=class_id,
            teacher_id=user_id,
            is_active=False,
        )

    async def clear_student_data_for_teacher(
        self,
        *,
        user_id: str,
        class_id: str,
        student_id: str,
        current_admin: UserModel,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AdminClearStudentDataResponse:
        """Reset all test submissions and grades for a student in teacher's class."""
        target_user = await self._user_repo.get_by_id(user_id)
        if not target_user:
            raise NotFoundError(
                message=f"User '{user_id}' not found",
                details={"user_id": user_id},
            )

        cls_obj = await self._class_repo.get_by_id(class_id)
        if not cls_obj or (cls_obj.teacher_uuid and cls_obj.teacher_uuid != user_id):
            raise NotFoundError(
                message=f"Class '{class_id}' not found for teacher '{user_id}'.",
                details={"class_id": class_id, "teacher_id": user_id},
            )

        student = await self._student_repo.get_by_id(student_id, class_id=class_id)
        if not student:
            raise NotFoundError(
                message=f"Student '{student_id}' not found in class '{class_id}'.",
                details={"student_id": student_id, "class_id": class_id},
            )

        deleted_count = await self._submission_repo.clear_student_submissions(
            student_id=student_id,
            class_id=class_id,
        )

        # Audit log
        await self._audit_repo.log_event(
            action="ADMIN_CLEAR_STUDENT_DATA",
            user_id=current_admin.id,
            user_email=current_admin.email,
            role=current_admin.role,
            target_entity="student",
            target_id=student_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={
                "admin_id": current_admin.id,
                "teacher_id": user_id,
                "class_id": class_id,
                "student_id": student_id,
                "deleted_submissions": deleted_count,
            },
        )

        # Live telemetry broadcast
        await telemetry_manager.broadcast({
            "action": "ADMIN_CLEAR_STUDENT_DATA",
            "user_id": user_id,
            "user_name": target_user.full_name,
            "school": target_user.school_name,
            "ip_address": ip_address,
            "details": {
                "admin_id": current_admin.id,
                "teacher_id": user_id,
                "class_id": class_id,
                "student_id": student_id,
                "deleted_submissions": deleted_count,
            },
        })

        return AdminClearStudentDataResponse(
            status="data_cleared",
            student_id=student_id,
            class_id=class_id,
            teacher_id=user_id,
            deleted_submissions_count=deleted_count,
        )

    async def query_audit_logs(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        action: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> AdminAuditLogsResponse:
        """Query system audit trail with multi-filter search and pagination."""
        logs, total = await self._audit_repo.query_logs(
            page=page,
            page_size=page_size,
            action=action,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )

        items: List[AdminAuditLogItem] = []
        for log in logs:
            details_dict = None
            if log.details_json:
                try:
                    details_dict = json.loads(log.details_json)
                except Exception:
                    details_dict = {"raw": log.details_json}

            items.append(
                AdminAuditLogItem(
                    id=log.id,
                    user_id=log.user_id,
                    user_email=log.user_email,
                    role=log.role,
                    action=log.action,
                    target_entity=log.target_entity,
                    target_id=log.target_id,
                    ip_address=log.ip_address,
                    user_agent=log.user_agent,
                    details=details_dict,
                    created_at=log.created_at,
                )
            )

        pages = (total + page_size - 1) // page_size if total > 0 else 1
        return AdminAuditLogsResponse(
            total=total,
            page=page,
            page_size=page_size,
            pages=pages,
            logs=items,
            items=items,
        )

    async def list_active_sessions(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
    ) -> AdminActiveSessionsResponse:
        """Platform-wide list of active user sessions."""
        skip = max(0, (page - 1) * page_size)
        sessions, total = await self._session_repo.list_all_active(skip=skip, limit=page_size)

        items: List[AdminSessionItem] = []
        for s in sessions:
            items.append(
                AdminSessionItem(
                    id=s.id,
                    user_id=s.user_id,
                    user_email=s.user.email if s.user else None,
                    user_full_name=s.user.full_name if s.user else None,
                    device_info=s.device_info,
                    ip_address=s.ip_address,
                    user_agent=s.user_agent,
                    created_at=s.created_at,
                    expires_at=s.expires_at,
                    revoked_at=s.revoked_at,
                )
            )

        pages = (total + page_size - 1) // page_size if total > 0 else 1
        return AdminActiveSessionsResponse(
            total=total,
            page=page,
            page_size=page_size,
            pages=pages,
            sessions=items,
            items=items,
        )

    async def kill_session(
        self,
        session_id: str,
        current_admin: UserModel,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AdminKillSessionResponse:
        """Force-terminate an active user session by session ID."""
        session_obj = await self._session_repo.get_by_id(session_id)
        if not session_obj or session_obj.revoked_at is not None:
            raise NotFoundError(
                message=f"Session '{session_id}' not found or already revoked",
                details={"session_id": session_id},
            )

        await self._session_repo.revoke_session(session_id)

        # Audit log
        await self._audit_repo.log_event(
            action="KILL_SESSION",
            user_id=current_admin.id,
            user_email=current_admin.email,
            role=current_admin.role,
            target_entity="session",
            target_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={"admin_id": current_admin.id, "target_user_id": session_obj.user_id},
        )

        # Live telemetry broadcast
        await telemetry_manager.broadcast({
            "action": "KILL_SESSION",
            "user_id": session_obj.user_id,
            "ip_address": ip_address,
            "details": {"admin_id": current_admin.id, "session_id": session_id},
        })

        return AdminKillSessionResponse(
            status="revoked",
            session_id=session_id,
            message="Session successfully terminated",
        )
