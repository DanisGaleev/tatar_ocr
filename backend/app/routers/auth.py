from typing import Any, Dict, Optional, Union
from fastapi import APIRouter, Depends, Header, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.dependencies import get_current_active_user
from app.core.exceptions import BusinessValidationError
from app.core.security import limiter
from app.models.user import UserModel
from app.services.auth_service import AuthService
from app.schemas.auth import (
    DeviceHandshakeRequest,
    DeviceHandshakeResponse,
)
from app.schemas.user import (
    UserRegisterRequest,
    UserRegisterResponse,
    VerifyEmailRequest,
    ResendCodeRequest,
    ResendCodeResponse,
    LoginRequest,
    UserResponse,
    TokenResponse,
    Login2FARequiredResponse,
    Verify2FARequest,
    Setup2FAResponse,
    Confirm2FARequest,
    Confirm2FAResponse,
    Disable2FARequest,
    RefreshTokenRequest,
    LogoutResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)

router = APIRouter(prefix="/auth", tags=["Authentication & Identity"])


def _get_auth_service(db: AsyncSession = Depends(get_db)) -> AuthService:
    return AuthService(db)


def _get_client_ip(request: Request) -> str:
    return request.client.host if request.client else "127.0.0.1"


# -----------------------------------------------------------------------------
# Mobile Device Handshake (Retained X-Teacher-UUID as install identifier only)
# -----------------------------------------------------------------------------
@router.post(
    "/device-handshake",
    response_model=DeviceHandshakeResponse,
    status_code=status.HTTP_200_OK,
    summary="Registers or verifies the teacher device UUID generated upon first install.",
)
async def device_handshake(
    payload: DeviceHandshakeRequest,
    x_teacher_uuid: str = Header(..., alias="X-Teacher-UUID"),
    auth_service: AuthService = Depends(_get_auth_service),
):
    return await auth_service.device_handshake(teacher_uuid=x_teacher_uuid, payload=payload)


# -----------------------------------------------------------------------------
# Registration & Email Verification
# -----------------------------------------------------------------------------
@router.post(
    "/register",
    response_model=UserRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new teacher account with email and password",
)
@limiter.limit("5/minute")
async def register_user(
    payload: UserRegisterRequest,
    request: Request,
    auth_service: AuthService = Depends(_get_auth_service),
):
    return await auth_service.register(
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name,
        school_name=payload.school_name,
    )


@router.post(
    "/verify-email",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify registration OTP code and activate account",
)
@limiter.limit("5/minute")
async def verify_email(
    payload: VerifyEmailRequest,
    request: Request,
    auth_service: AuthService = Depends(_get_auth_service),
):
    return await auth_service.verify_email(
        email=payload.email,
        code=payload.code,
        client_ip=_get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )


@router.post(
    "/resend-code",
    response_model=ResendCodeResponse,
    status_code=status.HTTP_200_OK,
    summary="Resend verification code with 60-second cooldown",
)
@limiter.limit("1/minute")
async def resend_code(
    payload: ResendCodeRequest,
    request: Request,
    auth_service: AuthService = Depends(_get_auth_service),
):
    return await auth_service.resend_code(
        email=payload.email,
        code_type=payload.code_type or "register_verify",
    )


# -----------------------------------------------------------------------------
# Login & 2FA Flow
# -----------------------------------------------------------------------------
@router.post(
    "/login",
    status_code=status.HTTP_200_OK,
    summary="Authenticate with email and password",
)
@limiter.limit("5/minute")
async def login(
    payload: LoginRequest,
    request: Request,
    auth_service: AuthService = Depends(_get_auth_service),
):
    result = await auth_service.login(
        email=payload.email,
        password=payload.password,
        client_ip=_get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
        device_info=payload.device_info,
    )
    if isinstance(result, Login2FARequiredResponse):
        return JSONResponse(status_code=status.HTTP_200_OK, content=result.model_dump())
    return result


@router.post(
    "/2fa/verify",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify 2FA code with temporary token and establish session",
)
@limiter.limit("5/minute")
async def verify_2fa(
    payload: Verify2FARequest,
    request: Request,
    auth_service: AuthService = Depends(_get_auth_service),
):
    return await auth_service.verify_2fa(
        temp_token=payload.temp_token,
        code=payload.code,
        client_ip=_get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )


@router.post(
    "/2fa/setup",
    response_model=Setup2FAResponse,
    status_code=status.HTTP_200_OK,
    summary="Initiate TOTP 2FA setup and get QR code",
)
async def setup_2fa(
    current_user: UserModel = Depends(get_current_active_user),
    auth_service: AuthService = Depends(_get_auth_service),
):
    return await auth_service.setup_2fa(current_user=current_user)


@router.post(
    "/2fa/confirm-setup",
    response_model=Confirm2FAResponse,
    status_code=status.HTTP_200_OK,
    summary="Confirm TOTP code and activate 2FA",
)
async def confirm_2fa_setup(
    payload: Confirm2FARequest,
    current_user: UserModel = Depends(get_current_active_user),
    auth_service: AuthService = Depends(_get_auth_service),
):
    return await auth_service.confirm_2fa_setup(current_user=current_user, code=payload.code)


@router.post(
    "/2fa/disable",
    status_code=status.HTTP_200_OK,
    summary="Disable two-factor authentication (requires password)",
)
async def disable_2fa(
    payload: Disable2FARequest,
    current_user: UserModel = Depends(get_current_active_user),
    auth_service: AuthService = Depends(_get_auth_service),
):
    return await auth_service.disable_2fa(current_user=current_user, password=payload.password)


# -----------------------------------------------------------------------------
# OAuth Integration (Yandex ID & VK ID)
# -----------------------------------------------------------------------------
@router.get(
    "/oauth/yandex/authorize",
    summary="Get Yandex ID OAuth authorization URL",
)
async def yandex_oauth_authorize():
    redirect_uri = "https://tatar-ocr.ru/auth/callback/yandex"
    url = f"https://oauth.yandex.ru/authorize?response_type=code&client_id={settings.YANDEX_CLIENT_ID}&redirect_uri={redirect_uri}"
    return {"authorization_url": url}


@router.post(
    "/oauth/yandex/callback",
    response_model=TokenResponse,
    summary="Exchange Yandex ID authorization code for session tokens",
)
async def yandex_oauth_callback(
    request: Request,
    payload: Dict[str, Any],
    auth_service: AuthService = Depends(_get_auth_service),
):
    code = payload.get("code")
    if not code:
        raise BusinessValidationError(message="Missing authorization code")
    return await auth_service.oauth_callback(
        provider="yandex",
        code=code,
        client_ip=_get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
        full_name=payload.get("full_name"),
    )


@router.get(
    "/oauth/vk/authorize",
    summary="Get VK ID OAuth authorization URL",
)
async def vk_oauth_authorize():
    redirect_uri = "https://tatar-ocr.ru/auth/callback/vk"
    url = f"https://id.vk.com/auth?app_id={settings.VK_CLIENT_ID}&response_type=code&redirect_uri={redirect_uri}"
    return {"authorization_url": url}


@router.post(
    "/oauth/vk/callback",
    response_model=TokenResponse,
    summary="Exchange VK ID authorization code for session tokens",
)
async def vk_oauth_callback(
    request: Request,
    payload: Dict[str, Any],
    auth_service: AuthService = Depends(_get_auth_service),
):
    code = payload.get("code")
    if not code:
        raise BusinessValidationError(message="Missing authorization code")
    return await auth_service.oauth_callback(
        provider="vk",
        code=code,
        client_ip=_get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
        full_name=payload.get("full_name"),
    )


# -----------------------------------------------------------------------------
# Token Rotation & Session Management
# -----------------------------------------------------------------------------
@router.post(
    "/refresh",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Rotate refresh token and issue new JWT access token",
)
async def refresh_tokens(
    payload: RefreshTokenRequest,
    request: Request,
    auth_service: AuthService = Depends(_get_auth_service),
):
    return await auth_service.refresh_tokens(
        refresh_token=payload.refresh_token,
        client_ip=_get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )


@router.post(
    "/logout",
    response_model=LogoutResponse,
    status_code=status.HTTP_200_OK,
    summary="Revoke active user sessions and log out",
)
async def logout(
    current_user: UserModel = Depends(get_current_active_user),
    auth_service: AuthService = Depends(_get_auth_service),
):
    return await auth_service.logout(current_user=current_user)


# -----------------------------------------------------------------------------
# Password Reset Flow
# -----------------------------------------------------------------------------
@router.post(
    "/forgot-password",
    status_code=status.HTTP_200_OK,
    summary="Request a 6-digit password reset code sent to email",
)
@limiter.limit("5/minute")
async def forgot_password(
    payload: ForgotPasswordRequest,
    request: Request,
    auth_service: AuthService = Depends(_get_auth_service),
):
    return await auth_service.forgot_password(email=payload.email)


@router.post(
    "/reset-password",
    status_code=status.HTTP_200_OK,
    summary="Reset password using OTP code",
)
@limiter.limit("5/minute")
async def reset_password(
    payload: ResetPasswordRequest,
    request: Request,
    auth_service: AuthService = Depends(_get_auth_service),
):
    return await auth_service.reset_password(
        email=payload.email,
        code=payload.code,
        new_password=payload.new_password,
    )


# -----------------------------------------------------------------------------
# Current User Profile
# -----------------------------------------------------------------------------
@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current authenticated user profile",
)
async def get_me(
    current_user: UserModel = Depends(get_current_active_user),
):
    return UserResponse.model_validate(current_user)
