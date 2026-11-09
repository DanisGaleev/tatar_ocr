from datetime import datetime, timezone
import json
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.core.exceptions import NotFoundError, PermissionDeniedError, BusinessValidationError
from app.models.user import UserModel
from app.models.school import ClassModel
from app.repositories.class_repo import ClassRepository
from app.repositories.student_repo import StudentRepository
from app.repositories.submission_repo import SubmissionRepository
from app.repositories.assignment_repo import AssignmentRepository
from app.generators.pdf_blank import render_batch_blanks_pdf
from app.schemas.classes import (
    ClassesListResponse,
    ClassItem,
    ClassStudentsResponse,
    StudentRosterItem,
    BulkImportStudentsResponse,
    ImportedStudentItem,
    AssignTestToClassRequest,
    ClassAssignmentItem,
    ClassAssignmentsResponse,
)


class ClassesService:
    """Service layer managing school classes, student rosters, bulk imports, and batch PDF blanks."""

    def __init__(self, db: AsyncSession):
        self._db: AsyncSession = db
        self._class_repo = ClassRepository(db)
        self._student_repo = StudentRepository(db)
        self._submission_repo = SubmissionRepository(db)
        self._assignment_repo = AssignmentRepository(db)

    async def list_classes(self, current_user: UserModel) -> ClassesListResponse:
        is_admin = current_user.role in ("admin", "superadmin")
        rows = await self._class_repo.list_classes_with_student_counts(
            teacher_uuid=current_user.id,
            is_admin=is_admin,
        )
        classes_out = [
            ClassItem(
                class_id=cls_obj.class_id,
                name=cls_obj.name,
                subject=cls_obj.subject,
                academic_year=cls_obj.academic_year,
                student_count=count,
            )
            for cls_obj, count in rows
        ]
        return ClassesListResponse(classes=classes_out)

    async def get_class_students(self, class_id: str, current_user: UserModel) -> ClassStudentsResponse:
        cls_obj = await self._get_verified_class(class_id, current_user)
        students = await self._student_repo.list_by_class(class_id, active_only=True)
        return ClassStudentsResponse(
            class_id=cls_obj.class_id,
            class_name=cls_obj.name,
            students=[
                StudentRosterItem(
                    student_id=s.student_id,
                    last_name=s.last_name,
                    first_name=s.first_name,
                    middle_name=s.middle_name or "",
                    full_name=s.full_name,
                    short_name=s.short_name,
                )
                for s in students
            ],
        )

    async def add_student(
        self,
        *,
        class_id: str,
        last_name: str,
        first_name: str,
        middle_name: Optional[str] = "",
        short_name: Optional[str] = None,
        current_user: UserModel,
    ) -> StudentRosterItem:
        await self._get_verified_class(class_id, current_user)
        if not last_name.strip() or not first_name.strip():
            raise BusinessValidationError(message="Last name and first name are required")

        student = await self._student_repo.create_student(
            class_id=class_id,
            last_name=last_name.strip(),
            first_name=first_name.strip(),
            middle_name=(middle_name or "").strip(),
            short_name=short_name.strip() if short_name else None,
        )
        return StudentRosterItem(
            student_id=student.student_id,
            last_name=student.last_name,
            first_name=student.first_name,
            middle_name=student.middle_name or "",
            full_name=student.full_name,
            short_name=student.short_name,
        )

    async def update_student(
        self,
        *,
        class_id: str,
        student_id: str,
        last_name: Optional[str] = None,
        first_name: Optional[str] = None,
        middle_name: Optional[str] = None,
        short_name: Optional[str] = None,
        current_user: UserModel,
    ) -> StudentRosterItem:
        await self._get_verified_class(class_id, current_user)
        student = await self._student_repo.get_by_id(student_id, class_id=class_id)
        if not student:
            raise NotFoundError(message=f"Student '{student_id}' not found.")

        updated = await self._student_repo.update_student(
            student,
            last_name=last_name,
            first_name=first_name,
            middle_name=middle_name,
            short_name=short_name,
        )
        return StudentRosterItem(
            student_id=updated.student_id,
            last_name=updated.last_name,
            first_name=updated.first_name,
            middle_name=updated.middle_name or "",
            full_name=updated.full_name,
            short_name=updated.short_name,
        )

    async def delete_student(
        self,
        *,
        class_id: str,
        student_id: str,
        current_user: UserModel,
    ) -> Dict[str, Any]:
        await self._get_verified_class(class_id, current_user)
        student = await self._student_repo.soft_delete(student_id, class_id=class_id)
        if not student:
            raise NotFoundError(message=f"Student '{student_id}' not found.")
        return {"status": "deleted", "student_id": student_id, "is_active": False}

    async def bulk_import_students(
        self,
        *,
        class_id: str,
        raw_text: str,
        current_user: UserModel,
    ) -> BulkImportStudentsResponse:
        cls_obj = await self._class_repo.get_by_id(class_id)
        if not cls_obj:
            cls_obj = await self._class_repo.create_class(
                class_id=class_id,
                name=class_id,
                teacher_uuid=current_user.id,
            )
        else:
            self._verify_class_access(cls_obj, current_user)

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return BulkImportStudentsResponse(added_count=0, students=[])

        items = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            ln = parts[0]
            fn = parts[1] if len(parts) > 1 else ""
            mn = parts[2] if len(parts) > 2 else ""
            items.append({
                "class_id": class_id,
                "last_name": ln,
                "first_name": fn,
                "middle_name": mn,
                "is_active": True,
            })

        created = await self._student_repo.bulk_create(items)
        return BulkImportStudentsResponse(
            added_count=len(created),
            students=[ImportedStudentItem(student_id=s.student_id, full_name=s.full_name) for s in created],
        )

    async def clear_student_data(
        self,
        *,
        class_id: str,
        student_id: str,
        current_user: UserModel,
    ) -> Dict[str, Any]:
        await self._get_verified_class(class_id, current_user)
        student = await self._student_repo.get_by_id(student_id, class_id=class_id)
        if not student:
            raise NotFoundError(message=f"Student '{student_id}' not found.")

        deleted_count = await self._submission_repo.clear_student_submissions(student_id, class_id=class_id)
        return {
            "status": "data_cleared",
            "student_id": student_id,
            "class_id": class_id,
            "deleted_submissions_count": deleted_count,
        }

    async def clear_class_students(
        self,
        *,
        class_id: str,
        current_user: UserModel,
    ) -> Dict[str, Any]:
        await self._get_verified_class(class_id, current_user)
        deleted_count = await self._student_repo.clear_class_students(class_id)
        return {"status": "cleared", "class_id": class_id, "deleted_students_count": deleted_count}

    async def generate_batch_blanks(
        self,
        *,
        class_id: str,
        assignment_id: str,
        current_user: UserModel,
    ) -> bytes:
        await self._get_verified_class(class_id, current_user)
        students = await self._student_repo.list_by_class(class_id, active_only=True)
        if not students:
            raise BusinessValidationError(
                message=f"No students in class '{class_id}' to generate batch blanks."
            )

        test_obj = await self._assignment_repo.get_by_id(assignment_id)
        if not test_obj:
            raise NotFoundError(message=f"Assignment '{assignment_id}' not found.")

        bundle = json.loads(test_obj.bundle_json)
        variants = bundle.get("variants", [])

        pdf_bytes = await run_in_threadpool(
            render_batch_blanks_pdf,
            assignment_id=assignment_id,
            title=test_obj.title,
            variants=variants,
            students=[{"student_id": s.student_id, "full_name": s.full_name} for s in students],
        )
        return pdf_bytes

    async def assign_test_to_class(
        self,
        *,
        class_id: str,
        payload: AssignTestToClassRequest,
        current_user: UserModel,
    ) -> ClassAssignmentItem:
        await self._get_verified_class(class_id, current_user)
        test_obj = await self._assignment_repo.get_by_id(payload.assignment_id)
        if not test_obj:
            raise NotFoundError(message=f"Assignment '{payload.assignment_id}' not found.")

        due_dt = None
        if payload.due_date:
            try:
                due_dt = datetime.fromisoformat(payload.due_date.replace("Z", "+00:00"))
            except Exception:
                pass

        ca_obj = await self._assignment_repo.assign_test_to_class(
            class_id=class_id,
            assignment_id=payload.assignment_id,
            due_date=due_dt,
            status=payload.status or "active",
        )
        total_students = await self._student_repo.count_by_class(class_id)

        return ClassAssignmentItem(
            id=ca_obj.id,
            class_id=class_id,
            assignment_id=payload.assignment_id,
            assignment_title=test_obj.title,
            total_variants=test_obj.total_variants,
            assigned_at=ca_obj.assigned_at.isoformat() if ca_obj.assigned_at else None,
            due_date=ca_obj.due_date.isoformat() if ca_obj.due_date else None,
            status=ca_obj.status,
            total_students=total_students,
            checked_submissions_count=0,
            pending_submissions_count=total_students,
            average_score_pct=0.0,
        )

    async def list_class_assignments(
        self,
        class_id: str,
        current_user: UserModel,
    ) -> ClassAssignmentsResponse:
        cls_obj = await self._get_verified_class(class_id, current_user)
        total_students = await self._student_repo.count_by_class(class_id)
        assignments = await self._assignment_repo.list_by_class(class_id)

        items = []
        for ca in assignments:
            test_obj = await self._assignment_repo.get_by_id(ca.assignment_id)
            title = test_obj.title if test_obj else ca.assignment_id
            variants_cnt = test_obj.total_variants if test_obj else 1

            subs = await self._submission_repo.list_by_assignment(ca.assignment_id)
            class_subs = [s for s in subs if s.class_id == class_id]
            checked_cnt = len(class_subs)
            pending_cnt = max(0, total_students - checked_cnt)
            avg_pct = (
                round(
                    sum(s.overall_score / s.max_score * 100.0 for s in class_subs if s.max_score > 0)
                    / checked_cnt,
                    1,
                )
                if checked_cnt > 0
                else 0.0
            )

            items.append(
                ClassAssignmentItem(
                    id=ca.id,
                    class_id=class_id,
                    assignment_id=ca.assignment_id,
                    assignment_title=title,
                    total_variants=variants_cnt,
                    assigned_at=ca.assigned_at.isoformat() if ca.assigned_at else None,
                    due_date=ca.due_date.isoformat() if ca.due_date else None,
                    status=ca.status,
                    total_students=total_students,
                    checked_submissions_count=checked_cnt,
                    pending_submissions_count=pending_cnt,
                    average_score_pct=avg_pct,
                )
            )

        return ClassAssignmentsResponse(
            class_id=class_id,
            class_name=cls_obj.name,
            assignments=items,
        )

    # -------------------------------------------------------------------------
    # Internal Helpers (Marked Private with Leading Underscore)
    # -------------------------------------------------------------------------
    async def _get_verified_class(self, class_id: str, current_user: UserModel) -> ClassModel:
        cls_obj = await self._class_repo.get_by_id(class_id)
        if not cls_obj:
            raise NotFoundError(message=f"Class '{class_id}' not found.", details={"class_id": class_id})
        self._verify_class_access(cls_obj, current_user)
        return cls_obj

    def _verify_class_access(self, cls_obj: ClassModel, current_user: UserModel) -> None:
        if current_user and current_user.role not in ("admin", "superadmin"):
            if not cls_obj.teacher_uuid or cls_obj.teacher_uuid != current_user.id:
                raise PermissionDeniedError(
                    message="Access denied: You do not own this class",
                    details={"class_id": cls_obj.class_id},
                )
