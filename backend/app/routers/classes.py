from typing import Optional
import uuid
from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.models.school import ClassModel, StudentModel
from app.schemas.classes import (
    ClassesListResponse,
    ClassItem,
    ClassStudentsResponse,
    StudentRosterItem,
    BulkImportStudentsRequest,
    BulkImportStudentsResponse,
    ImportedStudentItem,
)

router = APIRouter(prefix="/classes", tags=["Classes & Students"])

@router.get("", response_model=ClassesListResponse, summary="List all classes")
async def list_classes(
    x_teacher_uuid: Optional[str] = Header(None, alias="X-Teacher-UUID"),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(
        ClassModel,
        func.count(StudentModel.student_id).label("student_count"),
    ).outerjoin(
        StudentModel,
        ClassModel.class_id == StudentModel.class_id,
    ).group_by(ClassModel.class_id)

    if x_teacher_uuid:
        # Filter if teacher_uuid is set, or return all classes if teacher_uuid is None on class
        stmt = stmt.where((ClassModel.teacher_uuid == x_teacher_uuid) | (ClassModel.teacher_uuid.is_(None)))

    res = await db.execute(stmt)
    rows = res.all()

    classes_out = []
    for cls_obj, count in rows:
        classes_out.append(
            ClassItem(
                class_id=cls_obj.class_id,
                name=cls_obj.name,
                subject=cls_obj.subject,
                academic_year=cls_obj.academic_year,
                student_count=count,
            )
        )

    return ClassesListResponse(classes=classes_out)


@router.get("/{class_id}/students", response_model=ClassStudentsResponse, summary="Get student roster for class")
async def get_class_students(
    class_id: str,
    x_teacher_uuid: Optional[str] = Header(None, alias="X-Teacher-UUID"),
    db: AsyncSession = Depends(get_db),
):
    stmt_cls = select(ClassModel).where(ClassModel.class_id == class_id)
    res_cls = await db.execute(stmt_cls)
    cls_obj = res_cls.scalar_one_or_none()

    if not cls_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Class '{class_id}' not found.",
        )

    stmt_stu = select(StudentModel).where(StudentModel.class_id == class_id).order_by(StudentModel.last_name, StudentModel.first_name)
    res_stu = await db.execute(stmt_stu)
    students = res_stu.scalars().all()

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


@router.post(
    "/{class_id}/students/bulk-import",
    response_model=BulkImportStudentsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch import students from a pasted plain-text list",
)
async def bulk_import_students(
    class_id: str,
    payload: BulkImportStudentsRequest,
    x_teacher_uuid: Optional[str] = Header(None, alias="X-Teacher-UUID"),
    db: AsyncSession = Depends(get_db),
):
    stmt_cls = select(ClassModel).where(ClassModel.class_id == class_id)
    res_cls = await db.execute(stmt_cls)
    cls_obj = res_cls.scalar_one_or_none()

    if not cls_obj:
        # Auto-create class if it doesn't exist yet for seamless workflow
        cls_obj = ClassModel(
            class_id=class_id,
            name=class_id,
            teacher_uuid=x_teacher_uuid,
            subject="tatar_language",
            academic_year="2026-2027",
        )
        db.add(cls_obj)
        await db.commit()

    lines = [line.strip() for line in payload.raw_text.splitlines() if line.strip()]
    if not lines:
        return BulkImportStudentsResponse(added_count=0, students=[])

    imported_items = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        last_name = parts[0]
        first_name = parts[1] if len(parts) > 1 else ""
        middle_name = parts[2] if len(parts) > 2 else ""
        full_name = " ".join(parts)
        short_name = f"{last_name} {first_name[0]}." if first_name else last_name

        student_id = f"stu_{uuid.uuid4().hex[:6]}"
        student_obj = StudentModel(
            student_id=student_id,
            class_id=class_id,
            last_name=last_name,
            first_name=first_name,
            middle_name=middle_name,
            full_name=full_name,
            short_name=short_name,
        )
        db.add(student_obj)
        imported_items.append(ImportedStudentItem(student_id=student_id, full_name=full_name))

    await db.commit()

    return BulkImportStudentsResponse(
        added_count=len(imported_items),
        students=imported_items,
    )
