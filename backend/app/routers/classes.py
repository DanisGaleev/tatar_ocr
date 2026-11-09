import io
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import get_current_active_user
from app.models.user import UserModel
from app.services.classes_service import ClassesService
from app.schemas.classes import (
    ClassesListResponse,
    ClassStudentsResponse,
    StudentRosterItem,
    BulkImportStudentsRequest,
    BulkImportStudentsResponse,
    AssignTestToClassRequest,
    ClassAssignmentItem,
    ClassAssignmentsResponse,
)

router = APIRouter(prefix="/classes", tags=["Classes & Students"])


def _get_classes_service(db: AsyncSession = Depends(get_db)) -> ClassesService:
    return ClassesService(db)


class AddStudentRequest(BaseModel):
    last_name: str = Field(..., description="Last name (Фамилия)")
    first_name: str = Field(..., description="First name (Имя)")
    middle_name: Optional[str] = Field("", description="Middle name (Отчество)")
    short_name: Optional[str] = Field(None, description="Optional abbreviated name")


class UpdateStudentRequest(BaseModel):
    last_name: Optional[str] = None
    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    short_name: Optional[str] = None


@router.get("", response_model=ClassesListResponse, summary="List all classes")
async def list_classes(
    current_user: UserModel = Depends(get_current_active_user),
    classes_service: ClassesService = Depends(_get_classes_service),
):
    return await classes_service.list_classes(current_user)


@router.get("/{class_id}/students", response_model=ClassStudentsResponse, summary="Get student roster for class")
async def get_class_students(
    class_id: str,
    current_user: UserModel = Depends(get_current_active_user),
    classes_service: ClassesService = Depends(_get_classes_service),
):
    return await classes_service.get_class_students(class_id, current_user)


@router.post(
    "/{class_id}/students/bulk-import",
    response_model=BulkImportStudentsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch import students from a pasted plain-text list",
)
async def bulk_import_students(
    class_id: str,
    payload: BulkImportStudentsRequest,
    current_user: UserModel = Depends(get_current_active_user),
    classes_service: ClassesService = Depends(_get_classes_service),
):
    return await classes_service.bulk_import_students(
        class_id=class_id,
        raw_text=payload.raw_text,
        current_user=current_user,
    )


@router.post(
    "/{class_id}/students",
    response_model=StudentRosterItem,
    status_code=status.HTTP_201_CREATED,
    summary="Add a single student to a class",
)
async def add_single_student(
    class_id: str,
    payload: AddStudentRequest,
    current_user: UserModel = Depends(get_current_active_user),
    classes_service: ClassesService = Depends(_get_classes_service),
):
    return await classes_service.add_student(
        class_id=class_id,
        last_name=payload.last_name,
        first_name=payload.first_name,
        middle_name=payload.middle_name,
        short_name=payload.short_name,
        current_user=current_user,
    )


@router.patch(
    "/{class_id}/students/{student_id}",
    response_model=StudentRosterItem,
    status_code=status.HTTP_200_OK,
    summary="Update student names or details",
)
async def update_student(
    class_id: str,
    student_id: str,
    payload: UpdateStudentRequest,
    current_user: UserModel = Depends(get_current_active_user),
    classes_service: ClassesService = Depends(_get_classes_service),
):
    return await classes_service.update_student(
        class_id=class_id,
        student_id=student_id,
        last_name=payload.last_name,
        first_name=payload.first_name,
        middle_name=payload.middle_name,
        short_name=payload.short_name,
        current_user=current_user,
    )


@router.delete(
    "/{class_id}/students/{student_id}",
    status_code=status.HTTP_200_OK,
    summary="Soft-delete or deactivate student from class",
)
async def delete_student(
    class_id: str,
    student_id: str,
    current_user: UserModel = Depends(get_current_active_user),
    classes_service: ClassesService = Depends(_get_classes_service),
):
    return await classes_service.delete_student(
        class_id=class_id,
        student_id=student_id,
        current_user=current_user,
    )


@router.post(
    "/{class_id}/students/{student_id}/clear-data",
    status_code=status.HTTP_200_OK,
    summary="Clear all submission grades and history for a student",
)
async def clear_student_data(
    class_id: str,
    student_id: str,
    current_user: UserModel = Depends(get_current_active_user),
    classes_service: ClassesService = Depends(_get_classes_service),
):
    return await classes_service.clear_student_data(
        class_id=class_id,
        student_id=student_id,
        current_user=current_user,
    )


@router.post(
    "/{class_id}/assignments",
    response_model=ClassAssignmentItem,
    status_code=status.HTTP_201_CREATED,
    summary="Assign a test to a class",
)
async def assign_test_to_class(
    class_id: str,
    payload: AssignTestToClassRequest,
    current_user: UserModel = Depends(get_current_active_user),
    classes_service: ClassesService = Depends(_get_classes_service),
):
    return await classes_service.assign_test_to_class(
        class_id=class_id,
        payload=payload,
        current_user=current_user,
    )


@router.get(
    "/{class_id}/assignments",
    response_model=ClassAssignmentsResponse,
    summary="List tests assigned to a class with progress",
)
async def list_class_assignments(
    class_id: str,
    current_user: UserModel = Depends(get_current_active_user),
    classes_service: ClassesService = Depends(_get_classes_service),
):
    return await classes_service.list_class_assignments(
        class_id=class_id,
        current_user=current_user,
    )


@router.get(
    "/{class_id}/assignments/{assignment_id}/batch-blanks.pdf",
    summary="Download Batch Printable Blanks PDF for Class",
    description="Generates a multi-page PDF containing personalized test blanks for all students in the class.",
)
async def get_class_batch_blanks_pdf(
    class_id: str,
    assignment_id: str,
    current_user: UserModel = Depends(get_current_active_user),
    classes_service: ClassesService = Depends(_get_classes_service),
):
    pdf_bytes = await classes_service.generate_batch_blanks(
        class_id=class_id,
        assignment_id=assignment_id,
        current_user=current_user,
    )
    filename = f"batch_blanks_{class_id}_{assignment_id}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
