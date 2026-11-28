from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Header, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.teacher_scope import normalize_teacher_uuid
from app.models.teacher import TeacherModel
from app.schemas.auth import (
    DeviceHandshakeRequest,
    DeviceHandshakeResponse,
    TeacherPreferences,
    GradingScale,
)

router = APIRouter(prefix="/auth", tags=["Teacher Identity"])

@router.post(
    "/device-handshake",
    response_model=DeviceHandshakeResponse,
    status_code=status.HTTP_200_OK,
    summary="Registers or verifies the teacher device UUID generated upon first install.",
)
async def device_handshake(
    payload: DeviceHandshakeRequest,
    x_teacher_uuid: str = Header(..., alias="X-Teacher-UUID"),
    db: AsyncSession = Depends(get_db),
):
    teacher_uuid = normalize_teacher_uuid(x_teacher_uuid, required=True)
    stmt = select(TeacherModel).where(TeacherModel.teacher_uuid == teacher_uuid)
    res = await db.execute(stmt)
    teacher = res.scalar_one_or_none()

    if not teacher:
        teacher = TeacherModel(
            teacher_uuid=teacher_uuid,
            device_os=payload.device_os,
            app_version=payload.app_version,
            teacher_name=payload.teacher_name,
            school_name=payload.school_name,
            grade_5_min_pct=payload.grading_scale.grade_5_min_pct,
            grade_4_min_pct=payload.grading_scale.grade_4_min_pct,
            grade_3_min_pct=payload.grading_scale.grade_3_min_pct,
            confidence_flag_threshold=(
                payload.confidence_flag_threshold if payload.confidence_flag_threshold is not None else 0.65
            ),
        )
        db.add(teacher)
    else:
        teacher.device_os = payload.device_os
        teacher.app_version = payload.app_version
        teacher.teacher_name = payload.teacher_name
        teacher.school_name = payload.school_name
        teacher.grade_5_min_pct = payload.grading_scale.grade_5_min_pct
        teacher.grade_4_min_pct = payload.grading_scale.grade_4_min_pct
        teacher.grade_3_min_pct = payload.grading_scale.grade_3_min_pct
        if payload.confidence_flag_threshold is not None:
            teacher.confidence_flag_threshold = payload.confidence_flag_threshold
        teacher.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(teacher)

    return DeviceHandshakeResponse(
        status="active",
        teacher_uuid=teacher.teacher_uuid,
        teacher_name=teacher.teacher_name,
        preferences=TeacherPreferences(
            grading_scale=GradingScale(
                grade_5_min_pct=teacher.grade_5_min_pct,
                grade_4_min_pct=teacher.grade_4_min_pct,
                grade_3_min_pct=teacher.grade_3_min_pct,
            ),
            confidence_flag_threshold=teacher.confidence_flag_threshold,
        ),
    )
