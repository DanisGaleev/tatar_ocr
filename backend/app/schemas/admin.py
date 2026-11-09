from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.auth import TeacherPreferences
from app.schemas.classes import ClassItem, StudentRosterItem


class AdminUserItem(BaseModel):
    id: str
    email: str
    full_name: str
    school_name: Optional[str] = None
    role: str = "teacher"
    status: str = "active"
    email_verified: bool = True
    two_factor_enabled: bool = False
    two_factor_method: str = "none"
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AdminUserListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int = 1
    users: List[AdminUserItem] = Field(default_factory=list)
    items: List[AdminUserItem] = Field(default_factory=list)


class AdminUserStats(BaseModel):
    classes_count: int = 0
    students_count: int = 0
    assignments_count: int = 0
    submissions_count: int = 0
    active_sessions_count: int = 0
    average_grade: float = 0.0
    last_login_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None


class AdminSessionItem(BaseModel):
    id: str
    user_id: str
    user_email: Optional[str] = None
    user_full_name: Optional[str] = None
    device_info: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AdminUserDetailResponse(BaseModel):
    user: AdminUserItem
    stats: AdminUserStats
    recent_sessions: List[AdminSessionItem] = Field(default_factory=list)
    classes: List[ClassItem] = Field(default_factory=list)
    preferences: Optional[TeacherPreferences] = None


class AdminFreezeUserRequest(BaseModel):
    reason: Optional[str] = Field(None, description="Reason for account freeze")


class AdminFreezeResponse(BaseModel):
    status: str = "frozen"
    user_id: str
    sessions_revoked: int = 0
    message: str = "User frozen and all active sessions revoked"
    reason: Optional[str] = None


class AdminActionResponse(BaseModel):
    status: str
    user_id: Optional[str] = None
    message: str
    details: Optional[Dict[str, Any]] = None


class AdminResetPasswordRequest(BaseModel):
    new_password: Optional[str] = Field(
        None,
        description="New password (min 8 chars). If omitted, a secure temporary password is generated.",
    )
    send_email: bool = Field(False, description="Whether to send notification email")


class AdminResetPasswordResponse(BaseModel):
    status: str = "password_reset"
    user_id: str
    temporary_password: Optional[str] = None
    new_password: Optional[str] = None
    message: str = "Password successfully reset and sessions revoked"


class AdminUpdateProfileRequest(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2)
    school_name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None


class AdminClearDataResponse(BaseModel):
    status: str = "data_cleared"
    user_id: str
    deleted_classes_count: int = 0
    deleted_submissions_count: int = 0
    deleted_assignments_count: int = 0
    deleted_students_count: int = 0
    classes_cleared: int = 0
    submissions_cleared: int = 0
    message: str = "Teacher data cleared successfully"


class AddStudentRequest(BaseModel):
    last_name: str = Field(..., min_length=1, description="Student last name")
    first_name: str = Field(..., min_length=1, description="Student first name")
    middle_name: Optional[str] = Field("", description="Student middle name / patronymic")
    short_name: Optional[str] = Field(None, description="Optional abbreviated name")
    student_id: Optional[str] = Field(None, description="Optional custom student ID")


class AdminDeleteStudentResponse(BaseModel):
    status: str = "deleted"
    student_id: str
    class_id: str
    teacher_id: str
    is_active: bool = False


class AdminClearStudentDataResponse(BaseModel):
    status: str = "data_cleared"
    student_id: str
    class_id: str
    teacher_id: str
    deleted_submissions_count: int = 0


class AdminAuditLogItem(BaseModel):
    id: str
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    role: Optional[str] = None
    action: str
    target_entity: Optional[str] = None
    target_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AdminAuditLogsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int = 1
    logs: List[AdminAuditLogItem] = Field(default_factory=list)
    items: List[AdminAuditLogItem] = Field(default_factory=list)


class AdminActiveSessionsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int = 1
    sessions: List[AdminSessionItem] = Field(default_factory=list)
    items: List[AdminSessionItem] = Field(default_factory=list)


class AdminKillSessionResponse(BaseModel):
    status: str = "revoked"
    session_id: str
    message: str = "Session successfully terminated"


class AdminLiveFeedEvent(BaseModel):
    event_id: str
    timestamp: str
    action: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    school: Optional[str] = None
    ip_address: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Compatibility Aliases
AdminUserItemResponse = AdminUserItem
AdminUsersListResponse = AdminUserListResponse
AdminUserStatusResponse = AdminActionResponse
AdminUpdateUserProfileRequest = AdminUpdateProfileRequest
AdminCreateStudentRequest = AddStudentRequest
AuditLogItemResponse = AdminAuditLogItem
AuditLogsListResponse = AdminAuditLogsResponse
ActiveSessionItemResponse = AdminSessionItem
ActiveSessionsListResponse = AdminActiveSessionsResponse
RevokeSessionResponse = AdminKillSessionResponse
