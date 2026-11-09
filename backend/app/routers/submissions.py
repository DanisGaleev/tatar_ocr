from typing import Optional, List
from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import get_current_active_user
from app.core.security import limiter
from app.models.user import UserModel
from app.services.submission_service import SubmissionService
from app.schemas.submission import (
    BatchSyncSubmissionsRequest,
    BatchSyncSubmissionsResponse,
    SingleSubmissionSaveRequest,
    SingleSubmissionSaveResponse,
    SubmissionItem,
)

router = APIRouter(prefix="/submissions", tags=["Submissions Ingestion"])


def _get_submission_service(db: AsyncSession = Depends(get_db)) -> SubmissionService:
    return SubmissionService(db)


@router.post(
    "/batch-sync",
    response_model=BatchSyncSubmissionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch Sync Graded Submissions (Zero-Photo Class Upload)",
)
@limiter.limit("30/minute")
async def batch_sync_submissions(
    payload: BatchSyncSubmissionsRequest,
    request: Request,
    current_user: UserModel = Depends(get_current_active_user),
    submission_service: SubmissionService = Depends(_get_submission_service),
):
    return await submission_service.batch_sync_submissions(payload, current_user)


@router.post(
    "",
    response_model=SingleSubmissionSaveResponse,
    status_code=status.HTTP_200_OK,
    summary="Save Single Graded Submission",
)
async def save_single_submission(
    payload: SingleSubmissionSaveRequest,
    current_user: UserModel = Depends(get_current_active_user),
    submission_service: SubmissionService = Depends(_get_submission_service),
):
    return await submission_service.save_single_submission(payload, current_user)


@router.get(
    "",
    response_model=List[SubmissionItem],
    summary="Query Submissions",
)
async def query_submissions(
    class_id: Optional[str] = Query(None, description="Filter by class ID"),
    assignment_id: Optional[str] = Query(None, description="Filter by assignment ID"),
    student_id: Optional[str] = Query(None, description="Filter by student ID"),
    current_user: UserModel = Depends(get_current_active_user),
    submission_service: SubmissionService = Depends(_get_submission_service),
):
    return await submission_service.list_submissions(
        class_id=class_id,
        assignment_id=assignment_id,
        student_id=student_id,
        current_user=current_user,
    )
