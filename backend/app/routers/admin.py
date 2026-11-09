from datetime import datetime, timezone
import json
import logging
from typing import Optional
from fastapi import APIRouter, Depends, Path, Query, Request, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal, get_db
from app.core.dependencies import require_admin
from app.core.security import decode_token
from app.core.telemetry import telemetry_manager
from app.models.user import UserModel
from app.repositories.user_repo import UserRepository
from app.schemas.admin import (
    AddStudentRequest,
    AdminActionResponse,
    AdminActiveSessionsResponse,
    AdminAuditLogsResponse,
    AdminClearDataResponse,
    AdminClearStudentDataResponse,
    AdminDeleteStudentResponse,
    AdminFreezeResponse,
    AdminFreezeUserRequest,
    AdminKillSessionResponse,
    AdminResetPasswordRequest,
    AdminResetPasswordResponse,
    AdminUpdateProfileRequest,
    AdminUserDetailResponse,
    AdminUserItem,
    AdminUserListResponse,
)
from app.schemas.classes import StudentRosterItem
from app.services.admin_service import AdminService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin Management"])


def _extract_client_meta(request: Request):
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    return ip_address, user_agent


# -----------------------------------------------------------------------------
# 1. User Management: List Users
# -----------------------------------------------------------------------------
@router.get(
    "/users",
    response_model=AdminUserListResponse,
    summary="List platform users with filtering and pagination",
)
async def list_users(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Status filter: active, frozen, unverified, deleted"),
    role: Optional[str] = Query(None, description="Role filter: teacher, admin, superadmin"),
    search: Optional[str] = Query(None, description="Search term for full_name, email, school_name"),
    current_admin: UserModel = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminUserListResponse:
    service = AdminService(db)
    return await service.list_users(
        page=page,
        page_size=page_size,
        status=status,
        role=role,
        search=search,
    )


# -----------------------------------------------------------------------------
# 2. User Management: Get User Detail
# -----------------------------------------------------------------------------
@router.get(
    "/users/{user_id}",
    response_model=AdminUserDetailResponse,
    summary="Get user detail card with aggregated stats and sessions",
)
async def get_user_detail(
    user_id: str = Path(..., description="Target user ID"),
    current_admin: UserModel = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminUserDetailResponse:
    service = AdminService(db)
    return await service.get_user_detail(user_id=user_id)


# -----------------------------------------------------------------------------
# 3. Security: Freeze User
# -----------------------------------------------------------------------------
@router.post(
    "/users/{user_id}/freeze",
    response_model=AdminFreezeResponse,
    summary="Freeze user account and revoke all sessions immediately",
)
async def freeze_user(
    request: Request,
    user_id: str = Path(..., description="Target user ID"),
    payload: Optional[AdminFreezeUserRequest] = None,
    current_admin: UserModel = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminFreezeResponse:
    ip, ua = _extract_client_meta(request)
    service = AdminService(db)
    reason = payload.reason if payload else None
    return await service.freeze_user(
        user_id=user_id,
        current_admin=current_admin,
        reason=reason,
        ip_address=ip,
        user_agent=ua,
    )


# -----------------------------------------------------------------------------
# 4. Security: Unfreeze User
# -----------------------------------------------------------------------------
@router.post(
    "/users/{user_id}/unfreeze",
    response_model=AdminActionResponse,
    summary="Unfreeze user account and restore active status",
)
async def unfreeze_user(
    request: Request,
    user_id: str = Path(..., description="Target user ID"),
    current_admin: UserModel = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminActionResponse:
    ip, ua = _extract_client_meta(request)
    service = AdminService(db)
    return await service.unfreeze_user(
        user_id=user_id,
        current_admin=current_admin,
        ip_address=ip,
        user_agent=ua,
    )


# -----------------------------------------------------------------------------
# 5. User Management: Soft Delete User
# -----------------------------------------------------------------------------
@router.delete(
    "/users/{user_id}",
    response_model=AdminActionResponse,
    summary="Soft delete user account, anonymize PII, and revoke sessions",
)
async def delete_user(
    request: Request,
    user_id: str = Path(..., description="Target user ID"),
    current_admin: UserModel = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminActionResponse:
    ip, ua = _extract_client_meta(request)
    service = AdminService(db)
    return await service.delete_user(
        user_id=user_id,
        current_admin=current_admin,
        ip_address=ip,
        user_agent=ua,
    )


# -----------------------------------------------------------------------------
# 6. Identity: Reset User Password
# -----------------------------------------------------------------------------
@router.post(
    "/users/{user_id}/reset-password",
    response_model=AdminResetPasswordResponse,
    summary="Reset user password manually or generate temporary password",
)
async def reset_password(
    request: Request,
    user_id: str = Path(..., description="Target user ID"),
    payload: Optional[AdminResetPasswordRequest] = None,
    current_admin: UserModel = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminResetPasswordResponse:
    ip, ua = _extract_client_meta(request)
    service = AdminService(db)
    req = payload or AdminResetPasswordRequest()
    return await service.reset_password(
        user_id=user_id,
        payload=req,
        current_admin=current_admin,
        ip_address=ip,
        user_agent=ua,
    )


# -----------------------------------------------------------------------------
# 7. Profile: Patch User Profile
# -----------------------------------------------------------------------------
@router.patch(
    "/users/{user_id}/profile",
    response_model=AdminUserItem,
    summary="Update user name, school, email, or role",
)
async def update_profile(
    request: Request,
    user_id: str = Path(..., description="Target user ID"),
    payload: AdminUpdateProfileRequest = ...,
    current_admin: UserModel = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminUserItem:
    ip, ua = _extract_client_meta(request)
    service = AdminService(db)
    return await service.update_profile(
        user_id=user_id,
        payload=payload,
        current_admin=current_admin,
        ip_address=ip,
        user_agent=ua,
    )


# -----------------------------------------------------------------------------
# 8. Data Management: Clear Teacher Data
# -----------------------------------------------------------------------------
@router.post(
    "/users/{user_id}/clear-data",
    response_model=AdminClearDataResponse,
    summary="Purge teacher classes, assignments, and submissions (account intact)",
)
async def clear_teacher_data(
    request: Request,
    user_id: str = Path(..., description="Target teacher user ID"),
    current_admin: UserModel = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminClearDataResponse:
    ip, ua = _extract_client_meta(request)
    service = AdminService(db)
    return await service.clear_teacher_data(
        user_id=user_id,
        current_admin=current_admin,
        ip_address=ip,
        user_agent=ua,
    )


# -----------------------------------------------------------------------------
# 9. Teacher Roster: Add Student to Class
# -----------------------------------------------------------------------------
@router.post(
    "/users/{user_id}/classes/{class_id}/students",
    response_model=StudentRosterItem,
    status_code=status.HTTP_201_CREATED,
    summary="Add student to teacher's class",
)
async def add_student(
    request: Request,
    user_id: str = Path(..., description="Teacher user ID"),
    class_id: str = Path(..., description="Class ID"),
    payload: AddStudentRequest = ...,
    current_admin: UserModel = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> StudentRosterItem:
    ip, ua = _extract_client_meta(request)
    service = AdminService(db)
    return await service.add_student_for_teacher(
        user_id=user_id,
        class_id=class_id,
        payload=payload,
        current_admin=current_admin,
        ip_address=ip,
        user_agent=ua,
    )


# -----------------------------------------------------------------------------
# 10. Teacher Roster: Delete Student from Class
# -----------------------------------------------------------------------------
@router.delete(
    "/users/{user_id}/classes/{class_id}/students/{student_id}",
    response_model=AdminDeleteStudentResponse,
    summary="Soft delete student from teacher's class",
)
async def delete_student(
    request: Request,
    user_id: str = Path(..., description="Teacher user ID"),
    class_id: str = Path(..., description="Class ID"),
    student_id: str = Path(..., description="Student ID"),
    current_admin: UserModel = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminDeleteStudentResponse:
    ip, ua = _extract_client_meta(request)
    service = AdminService(db)
    return await service.delete_student_for_teacher(
        user_id=user_id,
        class_id=class_id,
        student_id=student_id,
        current_admin=current_admin,
        ip_address=ip,
        user_agent=ua,
    )


# -----------------------------------------------------------------------------
# 11. Teacher Roster: Clear Student Data
# -----------------------------------------------------------------------------
@router.post(
    "/users/{user_id}/classes/{class_id}/students/{student_id}/clear-data",
    response_model=AdminClearStudentDataResponse,
    summary="Reset test submissions and grades for a student in class",
)
async def clear_student_data(
    request: Request,
    user_id: str = Path(..., description="Teacher user ID"),
    class_id: str = Path(..., description="Class ID"),
    student_id: str = Path(..., description="Student ID"),
    current_admin: UserModel = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminClearStudentDataResponse:
    ip, ua = _extract_client_meta(request)
    service = AdminService(db)
    return await service.clear_student_data_for_teacher(
        user_id=user_id,
        class_id=class_id,
        student_id=student_id,
        current_admin=current_admin,
        ip_address=ip,
        user_agent=ua,
    )


# -----------------------------------------------------------------------------
# 12. Monitoring: Query Audit Logs
# -----------------------------------------------------------------------------
@router.get(
    "/audit-logs",
    response_model=AdminAuditLogsResponse,
    summary="Query audit trail with multi-filter search and pagination",
)
async def query_audit_logs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    action: Optional[str] = Query(None, description="Action filter, e.g. LOGIN, FREEZE_USER"),
    user_id: Optional[str] = Query(None, description="Target user ID filter"),
    start_date: Optional[datetime] = Query(None, description="Start date filter in ISO 8601"),
    end_date: Optional[datetime] = Query(None, description="End date filter in ISO 8601"),
    current_admin: UserModel = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminAuditLogsResponse:
    service = AdminService(db)
    return await service.query_audit_logs(
        page=page,
        page_size=page_size,
        action=action,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )


# -----------------------------------------------------------------------------
# 13. Session Management: List Active Sessions
# -----------------------------------------------------------------------------
@router.get(
    "/active-sessions",
    response_model=AdminActiveSessionsResponse,
    summary="List active user sessions platform-wide",
)
async def list_active_sessions(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    current_admin: UserModel = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminActiveSessionsResponse:
    service = AdminService(db)
    return await service.list_active_sessions(
        page=page,
        page_size=page_size,
    )


# -----------------------------------------------------------------------------
# 14. Session Management: Kill Session
# -----------------------------------------------------------------------------
@router.delete(
    "/active-sessions/{session_id}",
    response_model=AdminKillSessionResponse,
    summary="Force-terminate an active user session",
)
async def kill_session(
    request: Request,
    session_id: str = Path(..., description="Target session ID"),
    current_admin: UserModel = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminKillSessionResponse:
    ip, ua = _extract_client_meta(request)
    service = AdminService(db)
    return await service.kill_session(
        session_id=session_id,
        current_admin=current_admin,
        ip_address=ip,
        user_agent=ua,
    )


# -----------------------------------------------------------------------------
# 15. Real-Time Telemetry: WebSocket Live Feed
# -----------------------------------------------------------------------------
@router.websocket("/ws/live-feed")
async def admin_live_feed(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    admin_token: Optional[str] = Query(None),
):
    """
    WebSocket live feed streaming administrative and platform events.
    Authenticates via query parameter `?token=` or `?admin_token=`, or Authorization header.
    Rejects unauthenticated or non-admin callers with code 1008 (Policy Violation).
    """
    auth_token = token or admin_token
    if not auth_token:
        # Check Authorization header as fallback
        auth_hdr = websocket.headers.get("authorization")
        if auth_hdr and auth_hdr.lower().startswith("bearer "):
            auth_token = auth_hdr[7:].strip()

    if not auth_token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Verify JWT access token
    try:
        payload = decode_token(auth_token)
    except Exception as exc:
        logger.warning(f"WebSocket live feed auth rejected (decode error): {exc}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    if payload.get("type") != "access":
        logger.warning("WebSocket live feed auth rejected: token type is not access")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    user_id = payload.get("sub")
    if not user_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Check user in database
    session_factory = getattr(websocket.app.state, "session_factory", AsyncSessionLocal)
    async with session_factory() as db:
        user_repo = UserRepository(db)
        user = await user_repo.get_by_id(user_id)
        if not user or user.status != "active" or user.role not in ("admin", "superadmin"):
            logger.warning(f"WebSocket live feed auth rejected for user {user_id}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    # Connect to broadcaster pool (connect() accepts the WebSocket)
    await telemetry_manager.connect(websocket)

    try:
        while True:
            raw_msg = await websocket.receive_text()
            try:
                data = json.loads(raw_msg)
                if isinstance(data, dict) and data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
            except Exception:
                pass
    except (WebSocketDisconnect, Exception) as exc:
        logger.debug(f"Admin WebSocket disconnected: {exc}")
    finally:
        telemetry_manager.disconnect(websocket)
