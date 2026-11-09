from typing import Dict
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import get_current_active_user
from app.models.user import UserModel
from app.services.teacher_service import TeacherService
from app.schemas.user import (
    TeacherProfileResponse,
    TeacherProfileUpdateRequest,
    TeacherChangePasswordRequest,
)

router = APIRouter(prefix="/teacher", tags=["Teacher Profile & Settings"])


def _get_teacher_service(db: AsyncSession = Depends(get_db)) -> TeacherService:
    return TeacherService(db)


@router.get(
    "/profile",
    response_model=TeacherProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Get teacher profile and grading preferences",
)
async def get_teacher_profile(
    current_user: UserModel = Depends(get_current_active_user),
    teacher_service: TeacherService = Depends(_get_teacher_service),
):
    return await teacher_service.get_profile(current_user)


@router.patch(
    "/profile",
    response_model=TeacherProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Update teacher profile (ФИО, school) and grading preferences",
)
async def update_teacher_profile(
    payload: TeacherProfileUpdateRequest,
    current_user: UserModel = Depends(get_current_active_user),
    teacher_service: TeacherService = Depends(_get_teacher_service),
):
    return await teacher_service.update_profile(current_user, payload)


@router.post(
    "/change-password",
    status_code=status.HTTP_200_OK,
    summary="Change teacher password (requires old password and 2FA if active)",
)
async def change_password(
    payload: TeacherChangePasswordRequest,
    current_user: UserModel = Depends(get_current_active_user),
    teacher_service: TeacherService = Depends(_get_teacher_service),
) -> Dict[str, str]:
    return await teacher_service.change_password(current_user, payload)
