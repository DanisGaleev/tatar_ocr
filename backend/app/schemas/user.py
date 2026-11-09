from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.auth import GradingScale, TeacherPreferences


class UserRegisterRequest(BaseModel):
    email: str = Field(..., description="Teacher or Admin email address", examples=["teacher@example.com"])
    password: str = Field(..., min_length=8, description="Account password (min 8 characters)")
    full_name: str = Field(..., min_length=2, description="Full name (ФИО)", examples=["Каримова Гөлнара Илдар кызы"])
    school_name: Optional[str] = Field(None, description="School name or number", examples=["Гимназия №2"])


class UserRegisterResponse(BaseModel):
    status: str = "unverified"
    user_id: str
    email: str
    message: str = "Verification code sent to email"


class VerifyEmailRequest(BaseModel):
    email: str = Field(..., description="Email address to verify")
    code: str = Field(..., min_length=6, max_length=6, description="6-digit verification code")


class ResendCodeRequest(BaseModel):
    email: str = Field(..., description="Email address to resend code to")
    code_type: Optional[str] = Field("register_verify", description="Type of verification code")


class ResendCodeResponse(BaseModel):
    status: str = "code_sent"
    message: str = "New verification code sent"
    cooldown_seconds: int = 60


class LoginRequest(BaseModel):
    email: str = Field(..., description="Account email")
    password: str = Field(..., description="Account password")
    device_info: Optional[str] = Field(None, description="Optional device details, e.g. 'Chrome on Windows'")


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    school_name: Optional[str] = None
    role: str = "teacher"
    status: str = "active"
    email_verified: bool = True
    two_factor_enabled: bool = False
    two_factor_method: str = "none"
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    user: Optional[UserResponse] = None


class Login2FARequiredResponse(BaseModel):
    requires_2fa: bool = True
    two_factor_method: str = "totp"
    temp_token: str
    message: str = "2FA required"


class Verify2FARequest(BaseModel):
    temp_token: str = Field(..., description="Temporary token received from login")
    code: str = Field(..., description="6-digit TOTP or Email OTP code")


class Setup2FAResponse(BaseModel):
    secret: str
    otpauth_url: str
    qr_code_base64: str


class Confirm2FARequest(BaseModel):
    code: str = Field(..., description="6-digit TOTP code from authenticator app")


class Confirm2FAResponse(BaseModel):
    status: str = "active"
    two_factor_enabled: bool = True
    two_factor_method: str = "totp"
    backup_codes: List[str] = Field(default_factory=list)


class Disable2FARequest(BaseModel):
    password: str = Field(..., description="Current password for verification")


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., description="Valid refresh token")


class LogoutResponse(BaseModel):
    status: str = "logged_out"
    message: str = "Successfully logged out"


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., description="Account email to reset password for")


class ResetPasswordRequest(BaseModel):
    email: str = Field(..., description="Account email")
    code: str = Field(..., min_length=6, max_length=6, description="6-digit reset code from email")
    new_password: str = Field(..., min_length=8, description="New password (min 8 characters)")


class TeacherProfileResponse(BaseModel):
    user: UserResponse
    preferences: TeacherPreferences


class TeacherProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    school_name: Optional[str] = None
    grading_scale: Optional[GradingScale] = None
    confidence_flag_threshold: Optional[float] = None


class TeacherChangePasswordRequest(BaseModel):
    old_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password (min 8 characters)")
    two_factor_code: Optional[str] = Field(None, description="Optional 2FA code if 2FA is active")
