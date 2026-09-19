import json
import uuid
from typing import List, Optional
import base64
from fastapi import APIRouter, Depends, HTTPException, Query, status, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from app.core.database import get_db
from app.core.config import settings
from app.models.task import TaskBankModel
from app.models.test import AssembledTestModel
from app.schemas.task import (
    TaskBankItem,
    CreateTaskRequest,
    GenerateTasksRequest,
    GenerateTasksResponse,
    VerifyAnswerRequest,
    VerifyAnswerResponse,
    CellStatusItem,
    ScanTaskResponse,
    ScanTaskRequestBase64,
)
from app.schemas.test import (
    AssembleTestRequest,
    ReadyTestItem,
    OfflineBundleResponse,
)
from app.generators.registry import registry
from app.generators.phonetics import clean_word, build_expected_cells
from app.generators.checker import verify_question_cells
from app.services.task_extractor import get_task_extractor_service


router = APIRouter(prefix="/constructor", tags=["Assignment Constructor"])

@router.get("/task-types", summary="List supported task generator types")
async def list_task_types():
    return {"task_types": registry.list_supported_types()}

@router.get("/tasks", response_model=List[TaskBankItem], summary="Search Task Bank")
async def search_task_bank(
    query: Optional[str] = Query(None, description="Search term in prompt or answer"),
    topic_tag: Optional[str] = Query(None),
    grade_level: Optional[int] = Query(None, ge=5, le=9),
    max_cells: Optional[int] = Query(None, le=12),
    scope: Optional[str] = Query("all", pattern="^(all|my_tasks)$"),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(TaskBankModel)
    if topic_tag:
        stmt = stmt.where(TaskBankModel.topic_tag == topic_tag)
    if grade_level:
        stmt = stmt.where(TaskBankModel.grade_level == grade_level)
    if max_cells:
        stmt = stmt.where(TaskBankModel.cell_count <= max_cells)
    if query:
        search_pattern = f"%{query}%"
        stmt = stmt.where(
            or_(
                TaskBankModel.prompt_tt.ilike(search_pattern),
                TaskBankModel.expected_answer.ilike(search_pattern),
            )
        )
    result = await db.execute(stmt)
    tasks = result.scalars().all()
    return tasks

@router.post("/tasks", response_model=TaskBankItem, status_code=status.HTTP_201_CREATED, summary="Create Custom Single Task")
async def create_custom_task(
    payload: CreateTaskRequest,
    db: AsyncSession = Depends(get_db),
):
    clean_ans = clean_word(payload.expected_answer).upper()
    if len(clean_ans) > payload.cell_count:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Expected answer '{clean_ans}' ({len(clean_ans)} chars) exceeds specified cell count ({payload.cell_count}).",
        )
        
    task_obj = TaskBankModel(
        task_id=f"tsk_{uuid.uuid4().hex[:8]}",
        prompt_tt=payload.prompt_tt.strip(),
        expected_answer=clean_ans,
        cell_count=payload.cell_count,
        grade_level=payload.grade_level,
        topic_tag=payload.topic_tag,
        topic_name_tt=payload.topic_name_tt,
        task_type=payload.task_type or "custom",
        is_public_in_bank=payload.is_public_in_bank,
    )
    db.add(task_obj)
    await db.commit()
    await db.refresh(task_obj)
    return task_obj

@router.post("/generate", response_model=GenerateTasksResponse, summary="Generate tasks deterministically using linguistic generator")
async def generate_tasks(
    payload: GenerateTasksRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        gen = registry.get_generator(payload.task_type)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    try:
        drafts = gen.generate_batch(
            count=payload.count,
            seed=payload.seed,
            grade_level=payload.grade_level,
            stems=payload.custom_stems,
            case_code=payload.case_code,
        )
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Generation error: {err}")

    tasks_out: List[TaskBankItem] = []
    for draft in drafts:
        task_id = f"tsk_{uuid.uuid4().hex[:8]}"
        task_item = TaskBankItem(
            task_id=task_id,
            prompt_tt=draft.prompt_tt,
            expected_answer=draft.expected_answer,
            cell_count=draft.cell_count,
            grade_level=draft.grade_level,
            topic_tag=draft.topic_tag,
            topic_name_tt=draft.topic_name_tt,
            is_public_in_bank=payload.save_to_bank,
        )
        tasks_out.append(task_item)
        
        if payload.save_to_bank:
            db_task = TaskBankModel(
                task_id=task_id,
                prompt_tt=draft.prompt_tt,
                expected_answer=draft.expected_answer,
                cell_count=draft.cell_count,
                grade_level=draft.grade_level,
                topic_tag=draft.topic_tag,
                topic_name_tt=draft.topic_name_tt,
                task_type=draft.task_type,
                is_public_in_bank=True,
            )
            db.add(db_task)

    if payload.save_to_bank:
        await db.commit()

    return GenerateTasksResponse(
        task_type=payload.task_type,
        seed=payload.seed,
        count=len(tasks_out),
        tasks=tasks_out,
    )

@router.post("/tests", response_model=OfflineBundleResponse, status_code=status.HTTP_201_CREATED, summary="Assemble Modular Test")
async def assemble_modular_test(
    payload: AssembleTestRequest,
    db: AsyncSession = Depends(get_db),
):
    assignment_id = f"TAT-2026-Q{uuid.uuid4().hex[:4].upper()}"

    if payload.task_items:
        if len(payload.task_items) > settings.MAX_QUESTIONS_PER_PAGE:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Selected {len(payload.task_items)} questions exceeds maximum A4 physical blank limit ({settings.MAX_QUESTIONS_PER_PAGE}).",
            )
            
        task_ids = [t.task_id for t in payload.task_items]
        stmt = select(TaskBankModel).where(TaskBankModel.task_id.in_(task_ids))
        res = await db.execute(stmt)
        found_tasks = {t.task_id: t for t in res.scalars().all()}
        
        missing_ids = [t_id for t_id in task_ids if t_id not in found_tasks]
        if missing_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tasks not found in bank: {missing_ids}",
            )
            
        # Build variants from selected tasks
        variants_data = []
        for v_idx in range(1, payload.generate_variants_count + 1):
            q_list = list(payload.task_items)
            if payload.shuffle_questions_in_variants and v_idx > 1:
                # Deterministic rotation for variants
                q_list = q_list[1:] + q_list[:1]
                
            questions = []
            for order_idx, ref in enumerate(q_list, start=1):
                t_obj = found_tasks[ref.task_id]
                questions.append({
                    "question_number": order_idx,
                    "marker_id": 10 + order_idx,
                    "prompt": t_obj.prompt_tt,
                    "topic_tag": t_obj.topic_tag,
                    "topic_name_tt": t_obj.topic_name_tt,
                    "cell_count": t_obj.cell_count,
                    "expected_answer": t_obj.expected_answer,
                    "expected_cells": build_expected_cells(t_obj.expected_answer, t_obj.cell_count),
                })
                
            qr_signature = json.dumps({
                "tid": assignment_id,
                "var": v_idx,
                "page": 1,
                "tot": 1,
                "n_q": len(questions),
            }, separators=(',', ':'))
            
            variants_data.append({
                "variant_id": v_idx,
                "qr_signature": qr_signature,
                "template_geometry": {
                    "format": "A4",
                    "corner_aruco_dict": settings.ARUCO_DICT,
                    "corner_aruco_ids": settings.CORNER_ARUCO_IDS,
                    "cell_dimensions_mm": {
                        "width": settings.CELL_WIDTH_MM,
                        "height": settings.CELL_HEIGHT_MM,
                    },
                },
                "questions": questions,
            })
            
        bundle_data = {
            "assignment_id": assignment_id,
            "title": payload.title,
            "total_variants": payload.generate_variants_count,
            "variants": variants_data,
        }
    else:
        # Generate automatically using linguistic generators
        bundle_data = registry.assemble_variants(
            assignment_id=assignment_id,
            title=payload.title,
            grade_level=payload.grade_level,
            variants_count=payload.generate_variants_count,
            seed=payload.seed,
            questions_per_variant=8,
            shuffle=payload.shuffle_questions_in_variants,
        )

    bundle_data["test_id"] = assignment_id
    bundle_data["variants_count"] = payload.generate_variants_count

    test_record = AssembledTestModel(
        test_id=assignment_id,
        title=payload.title,
        grade_level=payload.grade_level,
        total_variants=payload.generate_variants_count,
        bundle_json=json.dumps(bundle_data, ensure_ascii=False),
    )
    db.add(test_record)
    await db.commit()

    return bundle_data

@router.get("/ready-tests", response_model=List[ReadyTestItem], summary="Browse Ready-Made Complete Tests")
async def get_ready_tests(db: AsyncSession = Depends(get_db)):
    stmt = select(AssembledTestModel).order_by(AssembledTestModel.created_at.desc()).limit(20)
    res = await db.execute(stmt)
    records = res.scalars().all()
    
    return [
        ReadyTestItem(
            test_id=rec.test_id,
            title=rec.title,
            grade_level=rec.grade_level,
            questions_count=8,
        )
        for rec in records
    ]

@router.post("/verify-answer", response_model=VerifyAnswerResponse, summary="Verify student answer against expected answer")
async def verify_student_answer(payload: VerifyAnswerRequest):
    st_clean = payload.student_answer.strip().upper()
    student_cells = [
        {"cell_index": i, "predicted_char": st_clean[i] if i < len(st_clean) else " "}
        for i in range(payload.cell_count)
    ]
    res = verify_question_cells(
        question_number=1,
        marker_id=11,
        topic_tag=payload.topic_tag,
        expected_answer=payload.expected_answer,
        cell_count=payload.cell_count,
        student_cells=student_cells,
    )
    return VerifyAnswerResponse(
        is_correct=res.is_correct,
        points_earned=res.points_earned,
        written_word=res.written_word,
        cells=[
            CellStatusItem(
                cell_index=c.cell_index,
                expected_char=c.expected_char,
                predicted_char=c.predicted_char,
                status=c.status,
            )
            for c in res.cells
        ],
    )


@router.post("/scan-task", response_model=ScanTaskResponse, summary="Scan exercise image via OCR and classify/structure via YandexGPT")
async def scan_task_from_image(
    file: UploadFile = File(..., description="Image file (photo or screenshot) of textbook/workbook exercise"),
    save_to_bank: bool = Query(False, description="Persist directly into task bank if valid"),
    grade_level: Optional[int] = Query(None, ge=5, le=9, description="Optional grade level hint (5..9)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Analyzes an uploaded photo or screenshot using OCR & YandexGPT:
    1. Extracts text from image via OCR.
    2. Classifies task type against supported taxonomy (or flags unsupported tasks).
    3. Formats prompt, expected answer (<=12 chars), and physical cell count.
    4. Optionally saves to task bank if save_to_bank=True.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file content type '{file.content_type}'. Must be an image (JPEG, PNG, WEBP).",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    service = get_task_extractor_service()
    try:
        result = await service.extract_from_image(
            image_bytes=image_bytes,
            mime_type=file.content_type,
            grade_hint=grade_level,
            save_to_bank=save_to_bank,
            db=db,
        )
        return result
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image with AI task extractor: {err}",
        )


@router.post("/scan-task-base64", response_model=ScanTaskResponse, summary="Scan exercise from base64 image via OCR & YandexGPT")
async def scan_task_from_base64(
    payload: ScanTaskRequestBase64,
    db: AsyncSession = Depends(get_db),
):
    """
    Base64 variant for mobile apps or JSON-only clients.
    """
    try:
        image_bytes = base64.b64decode(payload.image_base64)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 payload: {err}",
        )

    service = get_task_extractor_service()
    try:
        result = await service.extract_from_image(
            image_bytes=image_bytes,
            mime_type=payload.mime_type,
            grade_hint=payload.grade_level,
            save_to_bank=payload.save_to_bank,
            db=db,
        )
        return result
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image with AI task extractor: {err}",
        )

