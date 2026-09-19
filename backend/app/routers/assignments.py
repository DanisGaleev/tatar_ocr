import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.models.test import AssembledTestModel
from app.schemas.test import OfflineBundleResponse

router = APIRouter(prefix="/assignments", tags=["Assignment Delivery"])

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

