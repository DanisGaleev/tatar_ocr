import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.models.test import AssembledTestModel
from app.schemas.test import OfflineBundleResponse

router = APIRouter(prefix="/assignments", tags=["Assignment Delivery"])

@router.get("", summary="List Teacher Assignments")
async def list_assignments(
    class_id: Optional[str] = Query(None, description="Filter assignments by class ID"),
    status: Optional[str] = Query(None, description="Filter assignments by status"),
    db: AsyncSession = Depends(get_db),
):
    from app.models.school import ClassModel, ClassAssignmentModel, StudentModel
    from app.models.submission import SubmissionModel
    from sqlalchemy import func

    stmt = select(AssembledTestModel).order_by(AssembledTestModel.created_at.desc())
    res = await db.execute(stmt)
    tests = res.scalars().all()

    results = []
    for t in tests:
        # Check linked classes
        ca_stmt = select(ClassAssignmentModel, ClassModel).join(
            ClassModel, ClassAssignmentModel.class_id == ClassModel.class_id
        ).where(ClassAssignmentModel.assignment_id == t.test_id)

        if class_id:
            ca_stmt = ca_stmt.where(ClassAssignmentModel.class_id == class_id)
        if status:
            ca_stmt = ca_stmt.where(ClassAssignmentModel.status == status)

        ca_res = await db.execute(ca_stmt)
        ca_rows = ca_res.all()

        if (class_id or status) and not ca_rows:
            continue

        assigned_classes_list = []
        for ca_obj, cls_obj in ca_rows:
            stu_stmt = select(func.count(StudentModel.student_id)).where(StudentModel.class_id == cls_obj.class_id)
            total_students = (await db.execute(stu_stmt)).scalar() or 0

            sub_stmt = select(func.count(SubmissionModel.submission_id)).where(
                (SubmissionModel.class_id == cls_obj.class_id) & (SubmissionModel.assignment_id == t.test_id)
            )
            checked_count = (await db.execute(sub_stmt)).scalar() or 0

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


@router.get(
    "/{assignment_id}/batch-blanks.pdf",
    summary="Download Batch Printable Blanks PDF (Alias)",
    description="Generates a multi-page PDF containing personalized test blanks for all students in the specified class.",
)
async def get_assignment_batch_blanks_pdf(
    assignment_id: str,
    class_id: str = Query(..., description="Class ID whose students to generate blanks for"),
    db: AsyncSession = Depends(get_db),
):
    from app.routers.classes import get_class_batch_blanks_pdf
    return await get_class_batch_blanks_pdf(class_id=class_id, assignment_id=assignment_id, db=db)


@router.get("/{assignment_id}/offline-bundle", response_model=OfflineBundleResponse, summary="Download Offline Verification Bundle")
async def get_offline_bundle(
    assignment_id: str,
    variant: Optional[int] = Query(None, description="Optional variant filter"),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(AssembledTestModel).where(AssembledTestModel.test_id == assignment_id)
    res = await db.execute(stmt)
    test_obj = res.scalar_one_or_none()
    
    if not test_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assignment '{assignment_id}' not found.",
        )

    bundle_dict = json.loads(test_obj.bundle_json)
    if variant is not None:
        filtered_variants = [v for v in bundle_dict.get("variants", []) if v.get("variant_id") == variant]
        if not filtered_variants:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Variant {variant} for assignment '{assignment_id}' not found.",
            )
        bundle_dict["variants"] = filtered_variants
        bundle_dict["total_variants"] = len(filtered_variants)
        
    return bundle_dict

@router.get("/{assignment_id}", summary="Get Assignment Metadata")
async def get_assignment_metadata(
    assignment_id: str,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(AssembledTestModel).where(AssembledTestModel.test_id == assignment_id)
    res = await db.execute(stmt)
    test_obj = res.scalar_one_or_none()
    
    if not test_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assignment '{assignment_id}' not found.",
        )
        
    return {
        "assignment_id": test_obj.test_id,
        "title": test_obj.title,
        "grade_level": test_obj.grade_level,
        "total_variants": test_obj.total_variants,
        "created_at": test_obj.created_at.isoformat(),
    }


@router.get(
    "/{assignment_id}/blank.pdf",
    summary="Download Printable Blank PDF (Generic or Single Student)",
    description="Generates a print-ready A4 test sheet with 4 corner ArUco markers, QR code, and 10x10 mm answer boxes.",
)
async def get_blank_pdf(
    assignment_id: str,
    variant: int = Query(1, description="Test variant number"),
    student_id: Optional[str] = Query(None, description="Pre-prints student name in header boxes"),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(AssembledTestModel).where(AssembledTestModel.test_id == assignment_id)
    res = await db.execute(stmt)
    test_obj = res.scalar_one_or_none()

    if not test_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assignment '{assignment_id}' not found.",
        )

    bundle_dict = json.loads(test_obj.bundle_json)
    variants = bundle_dict.get("variants", [])
    matching_variant = next((v for v in variants if v.get("variant_id") == variant), None)

    if not matching_variant:
        if variants:
            matching_variant = variants[0]
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Variant {variant} not found in assignment '{assignment_id}'.",
            )

    questions = matching_variant.get("questions", [])

    # Look up student if specified
    student_name = None
    if student_id:
        from app.models.school import StudentModel
        stmt_stu = select(StudentModel).where(StudentModel.student_id == student_id)
        res_stu = await db.execute(stmt_stu)
        stu_obj = res_stu.scalar_one_or_none()
        if stu_obj:
            student_name = stu_obj.full_name
        else:
            student_name = student_id

    import io
    from fastapi.responses import StreamingResponse
    from app.generators.pdf_blank import render_blank_pdf

    pdf_bytes = render_blank_pdf(
        assignment_id=assignment_id,
        title=test_obj.title,
        variant_id=variant,
        questions=questions,
        student_name=student_name,
    )

    filename = f"blank_{assignment_id}_var_{variant}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )

