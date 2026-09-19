from typing import Optional
import uuid
from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.models.school import ClassModel, StudentModel, ClassAssignmentModel
from app.models.test import AssembledTestModel
from app.models.submission import SubmissionModel
from app.schemas.classes import (
    ClassesListResponse,
    ClassItem,
    ClassStudentsResponse,
    StudentRosterItem,
    BulkImportStudentsRequest,
    BulkImportStudentsResponse,
    ImportedStudentItem,
    AssignTestToClassRequest,
    ClassAssignmentItem,
    ClassAssignmentsResponse,
)
from app.generators.pdf_blank import render_batch_blanks_pdf

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


@router.post(
    "/{class_id}/assignments",
    response_model=ClassAssignmentItem,
    status_code=status.HTTP_201_CREATED,
    summary="Assign a test to a class",
)
async def assign_test_to_class(
    class_id: str,
    payload: AssignTestToClassRequest,
    db: AsyncSession = Depends(get_db),
):
    stmt_cls = select(ClassModel).where(ClassModel.class_id == class_id)
    res_cls = await db.execute(stmt_cls)
    cls_obj = res_cls.scalar_one_or_none()
    if not cls_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Class '{class_id}' not found.")

    stmt_test = select(AssembledTestModel).where(AssembledTestModel.test_id == payload.assignment_id)
    res_test = await db.execute(stmt_test)
    test_obj = res_test.scalar_one_or_none()
    if not test_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Assignment '{payload.assignment_id}' not found.")

    # Check if already assigned
    stmt_ca = select(ClassAssignmentModel).where(
        (ClassAssignmentModel.class_id == class_id) & (ClassAssignmentModel.assignment_id == payload.assignment_id)
    )
    res_ca = await db.execute(stmt_ca)
    ca_obj = res_ca.scalar_one_or_none()

    from datetime import datetime
    due_dt = None
    if payload.due_date:
        try:
            due_dt = datetime.fromisoformat(payload.due_date.replace("Z", "+00:00"))
        except Exception:
            pass

    if not ca_obj:
        ca_obj = ClassAssignmentModel(
            id=f"ca_{uuid.uuid4().hex[:8]}",
            class_id=class_id,
            assignment_id=payload.assignment_id,
            due_date=due_dt,
            status=payload.status or "active",
        )
        db.add(ca_obj)
    else:
        if due_dt:
            ca_obj.due_date = due_dt
        if payload.status:
            ca_obj.status = payload.status

    await db.commit()
    await db.refresh(ca_obj)

    # Count students
    stu_stmt = select(func.count(StudentModel.student_id)).where(StudentModel.class_id == class_id)
    stu_cnt = (await db.execute(stu_stmt)).scalar() or 0

    return ClassAssignmentItem(
        id=ca_obj.id,
        class_id=class_id,
        assignment_id=payload.assignment_id,
        assignment_title=test_obj.title,
        total_variants=test_obj.total_variants,
        assigned_at=ca_obj.assigned_at.isoformat() if ca_obj.assigned_at else None,
        due_date=ca_obj.due_date.isoformat() if ca_obj.due_date else None,
        status=ca_obj.status,
        total_students=stu_cnt,
        checked_submissions_count=0,
        pending_submissions_count=stu_cnt,
        average_score_pct=0.0,
    )


@router.get(
    "/{class_id}/assignments",
    response_model=ClassAssignmentsResponse,
    summary="List tests assigned to a class with progress",
)
async def list_class_assignments(
    class_id: str,
    db: AsyncSession = Depends(get_db),
):
    stmt_cls = select(ClassModel).where(ClassModel.class_id == class_id)
    res_cls = await db.execute(stmt_cls)
    cls_obj = res_cls.scalar_one_or_none()
    if not cls_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Class '{class_id}' not found.")

    stu_stmt = select(func.count(StudentModel.student_id)).where(StudentModel.class_id == class_id)
    total_students = (await db.execute(stu_stmt)).scalar() or 0

    stmt_ca = select(ClassAssignmentModel).where(ClassAssignmentModel.class_id == class_id).order_by(ClassAssignmentModel.assigned_at.desc())
    res_ca = await db.execute(stmt_ca)
    assignments = res_ca.scalars().all()

    items = []
    for ca in assignments:
        stmt_test = select(AssembledTestModel).where(AssembledTestModel.test_id == ca.assignment_id)
        test_obj = (await db.execute(stmt_test)).scalar_one_or_none()
        title = test_obj.title if test_obj else ca.assignment_id
        variants_cnt = test_obj.total_variants if test_obj else 1

        # Check submissions progress
        sub_stmt = select(SubmissionModel).where(
            (SubmissionModel.class_id == class_id) & (SubmissionModel.assignment_id == ca.assignment_id)
        )
        subs = (await db.execute(sub_stmt)).scalars().all()
        checked_cnt = len(subs)
        pending_cnt = max(0, total_students - checked_cnt)
        avg_pct = round(sum(s.overall_score / s.max_score * 100.0 for s in subs if s.max_score > 0) / checked_cnt, 1) if checked_cnt > 0 else 0.0

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


@router.get(
    "/{class_id}/assignments/{assignment_id}/batch-blanks.pdf",
    summary="Download Batch Printable Blanks PDF for Class",
    description="Generates a multi-page PDF containing personalized test blanks for all students in the class.",
)
async def get_class_batch_blanks_pdf(
    class_id: str,
    assignment_id: str,
    db: AsyncSession = Depends(get_db),
):
    import io
    import json
    from fastapi.responses import StreamingResponse

    stmt_cls = select(ClassModel).where(ClassModel.class_id == class_id)
    cls_obj = (await db.execute(stmt_cls)).scalar_one_or_none()
    if not cls_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Class '{class_id}' not found.")

    stmt_stu = select(StudentModel).where(StudentModel.class_id == class_id).order_by(StudentModel.last_name, StudentModel.first_name)
    students = (await db.execute(stmt_stu)).scalars().all()
    if not students:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"No students in class '{class_id}' to generate batch blanks.")

    stmt_test = select(AssembledTestModel).where(AssembledTestModel.test_id == assignment_id)
    test_obj = (await db.execute(stmt_test)).scalar_one_or_none()
    if not test_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Assignment '{assignment_id}' not found.")

    bundle_dict = json.loads(test_obj.bundle_json)
    variants = bundle_dict.get("variants", [])

    pdf_bytes = render_batch_blanks_pdf(
        assignment_id=assignment_id,
        title=test_obj.title,
        variants=variants,
        students=[{"student_id": s.student_id, "full_name": s.full_name} for s in students],
    )

    filename = f"batch_blanks_{class_id}_{assignment_id}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )

