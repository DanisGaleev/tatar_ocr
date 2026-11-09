from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Union
import secrets

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.email import email_service
from app.core.exceptions import (
    AuthError,
    PermissionDeniedError,
    NotFoundError,
    BusinessValidationError,
    RateLimitError,
)
from app.core.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_refresh_token,
    generate_totp_secret,
    get_totp_uri,
    generate_qr_code_base64,
    verify_totp_code,
    generate_numeric_otp,
)
from app.core.telemetry import telemetry_manager
from app.models.user import UserModel
from app.repositories.user_repo import UserRepository
from app.repositories.session_repo import SessionRepository
from app.repositories.verification_repo import VerificationCodeRepository
from app.repositories.oauth_repo import OAuthRepository
from app.repositories.audit_repo import AuditRepository
from app.repositories.teacher_repo import TeacherRepository
from app.schemas.auth import DeviceHandshakeRequest, DeviceHandshakeResponse, TeacherPreferences, GradingScale
from app.schemas.user import (
    UserRegisterResponse,
    TokenResponse,
    UserResponse,
    Login2FARequiredResponse,
    Setup2FAResponse,
    Confirm2FAResponse,
    ResendCodeResponse,
    LogoutResponse,
)


class AuthService:
    """Service layer encapsulating identity, authentication, 2FA, OAuth, and session lifecycles."""

    def __init__(self, db: AsyncSession):
        self._db: AsyncSession = db
        self._user_repo = UserRepository(db)
        self._session_repo = SessionRepository(db)
        self._verification_repo = VerificationCodeRepository(db)
        self._oauth_repo = OAuthRepository(db)
        self._audit_repo = AuditRepository(db)
        self._teacher_repo = TeacherRepository(db)

    async def register(
        self,
        *,
        email: str,
        password: str,
        full_name: str,
        school_name: Optional[str] = None,
    ) -> UserRegisterResponse:
        norm_email = email.strip().lower()
        if len(password) < 8:
            raise BusinessValidationError(message="Password must be at least 8 characters long")

        existing_user = await self._user_repo.get_by_email(norm_email)
        if existing_user and existing_user.status in ["active", "frozen"]:
            raise BusinessValidationError(message="Email is already registered")

        hashed = get_password_hash(password)
        if existing_user and existing_user.status == "unverified":
            user = await self._user_repo.update_user(
                existing_user,
                {
                    "hashed_password": hashed,
                    "full_name": full_name.strip(),
                    "school_name": school_name.strip() if school_name else None,
                },
            )
        else:
            user = await self._user_repo.create_user(
                email=norm_email,
                hashed_password=hashed,
                full_name=full_name.strip(),
                school_name=school_name.strip() if school_name else None,
                role="teacher",
                status="unverified",
                email_verified=False,
            )

        await self._verification_repo.invalidate_all(norm_email, "register_verify")
        code = generate_numeric_otp(6)
        await self._verification_repo.create_code(
            email=norm_email,
            code=code,
            code_type="register_verify",
            ttl_minutes=10,
        )

        await email_service.send_verification_code(norm_email, code, code_type="register_verify")

        return UserRegisterResponse(
            status="unverified",
            user_id=user.id,
            email=user.email,
            message="Verification code sent to email",
        )

    async def verify_email(
        self,
        *,
        email: str,
        code: str,
        client_ip: str = "127.0.0.1",
        user_agent: Optional[str] = None,
    ) -> TokenResponse:
        norm_email = email.strip().lower()
        vcd = await self._verification_repo.get_valid_code(norm_email, "register_verify")
        if not vcd or vcd.attempts_left <= 0:
            raise BusinessValidationError(message="Invalid or expired verification code")

        if vcd.code != code.strip():
            remaining = await self._verification_repo.decrement_attempts(vcd)
            raise BusinessValidationError(message=f"Invalid verification code. Attempts remaining: {remaining}")

        await self._verification_repo.consume_code(vcd)

        user = await self._user_repo.get_by_email(norm_email)
        if not user:
            raise NotFoundError(message="User not found")

        await self._user_repo.update_user(user, {"status": "active", "email_verified": True})
        await self._user_repo.update_last_login(user)

        # Ensure linked teacher profile exists
        teacher = await self._teacher_repo.get_by_user_id(user.id)
        if not teacher:
            await self._teacher_repo.create_profile_for_user(user)

        session, access_token, refresh_token = await self._create_user_session(user, client_ip, user_agent)
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse.model_validate(user),
        )

    async def resend_code(
        self,
        *,
        email: str,
        code_type: str = "register_verify",
    ) -> ResendCodeResponse:
        norm_email = email.strip().lower()
        last_created = await self._verification_repo.get_last_created_at(norm_email, code_type)
        if last_created:
            now_utc = self._now_utc()
            created_utc = last_created.replace(tzinfo=timezone.utc) if last_created.tzinfo is None else last_created
            elapsed = (now_utc - created_utc).total_seconds()
            if elapsed < 60:
                retry_after = int(60 - elapsed)
                raise RateLimitError(
                    message=f"Please wait {retry_after} seconds before requesting a new code",
                    details={"retry_after": retry_after},
                )

        code = generate_numeric_otp(6)
        await self._verification_repo.create_code(
            email=norm_email,
            code=code,
            code_type=code_type,
            ttl_minutes=10,
        )
        await email_service.send_verification_code(norm_email, code, code_type=code_type)

        return ResendCodeResponse(
            status="code_sent",
            message="New verification code sent",
            cooldown_seconds=60,
        )

    async def login(
        self,
        *,
        email: str,
        password: str,
        client_ip: str = "127.0.0.1",
        user_agent: Optional[str] = None,
        device_info: Optional[str] = None,
    ) -> Union[TokenResponse, Login2FARequiredResponse]:
        norm_email = email.strip().lower()
        now_utc = self._now_utc()
        user = await self._user_repo.get_by_email(norm_email)

        if not user or user.status == "deleted":
            raise AuthError(message="Invalid email or password")
        if user.status == "frozen":
            raise PermissionDeniedError(message="Account is frozen")

        # Brute-force lockout verification
        if user.locked_until:
            locked_utc = user.locked_until.replace(tzinfo=timezone.utc) if user.locked_until.tzinfo is None else user.locked_until
            if locked_utc > now_utc:
                remaining = max(1, int((locked_utc - now_utc).total_seconds() / 60))
                raise PermissionDeniedError(
                    message=f"Account is locked due to too many failed attempts. Try again in {remaining} minutes."
                )
            await self._user_repo.reset_failed_attempts(user)

        # Password verification
        if not user.hashed_password or not verify_password(password, user.hashed_password):
            user = await self._user_repo.increment_failed_attempts(user)
            if user and user.failed_login_attempts >= 5:
                await self._audit_repo.log_event(
                    action="BRUTE_FORCE_LOCKOUT",
                    user_id=user.id,
                    user_email=user.email,
                    role=user.role,
                    target_entity="user",
                    target_id=user.id,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    details={"reason": "5 consecutive failed login attempts", "lock_minutes": 15},
                )
                await telemetry_manager.broadcast({
                    "action": "BRUTE_FORCE_LOCKOUT",
                    "user_id": user.id,
                    "user_name": user.full_name,
                    "school": user.school_name or "",
                    "ip_address": client_ip,
                    "details": {"reason": "5 consecutive failed login attempts"},
                })
                raise PermissionDeniedError(
                    message="Account is locked due to too many failed attempts. Try again in 15 minutes."
                )
            raise AuthError(message="Invalid email or password")

        await self._user_repo.reset_failed_attempts(user)

        if user.status == "unverified":
            raise PermissionDeniedError(message="Account email is not verified. Please verify your email first.")

        # 2FA Check
        if user.two_factor_enabled:
            temp_token = create_access_token(
                {"sub": user.id, "email": user.email, "role": user.role, "type": "2fa_temp"},
                expires_delta=timedelta(minutes=10),
            )
            if user.two_factor_method == "email":
                otp_code = generate_numeric_otp(6)
                await self._verification_repo.create_code(
                    email=user.email,
                    code=otp_code,
                    code_type="login_2fa",
                    ttl_minutes=10,
                )
                await email_service.send_2fa_code(user.email, otp_code)

            return Login2FARequiredResponse(
                requires_2fa=True,
                two_factor_method=user.two_factor_method,
                temp_token=temp_token,
                message=f"2FA required via {user.two_factor_method}",
            )

        # Standard successful login
        await self._user_repo.update_last_login(user)
        session, access_token, refresh_token = await self._create_user_session(
            user, client_ip, user_agent, device_info=device_info
        )
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse.model_validate(user),
        )

    async def verify_2fa(
        self,
        *,
        temp_token: str,
        code: str,
        client_ip: str = "127.0.0.1",
        user_agent: Optional[str] = None,
    ) -> TokenResponse:
        try:
            token_data = decode_token(temp_token)
        except Exception:
            raise AuthError(message="Invalid 2FA temporary token")

        if token_data.get("type") != "2fa_temp":
            raise AuthError(message="Invalid 2FA temporary token")

        user_id = token_data.get("sub")
        if not user_id:
            raise AuthError(message="Invalid token payload")

        user = await self._user_repo.get_by_id(user_id)
        if not user or user.status == "frozen":
            raise PermissionDeniedError(message="Account is frozen or invalid")

        if user.two_factor_method == "totp":
            if not user.totp_secret or not verify_totp_code(user.totp_secret, code):
                raise BusinessValidationError(message="Invalid 2FA code")
        elif user.two_factor_method == "email":
            vcd = await self._verification_repo.get_valid_code(user.email, "login_2fa")
            if not vcd or vcd.attempts_left <= 0:
                raise BusinessValidationError(message="Invalid or expired 2FA code")
            if vcd.code != code.strip():
                remaining = await self._verification_repo.decrement_attempts(vcd)
                raise BusinessValidationError(message=f"Invalid 2FA code. Attempts left: {remaining}")
            await self._verification_repo.consume_code(vcd)
        else:
            raise BusinessValidationError(message="2FA is not enabled for this account")

        await self._user_repo.update_last_login(user)
        session, access_token, refresh_token = await self._create_user_session(user, client_ip, user_agent)
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse.model_validate(user),
        )

    async def setup_2fa(self, current_user: UserModel) -> Setup2FAResponse:
        secret = generate_totp_secret()
        await self._user_repo.set_2fa_status(
            current_user,
            enabled=False,
            method="totp",
            totp_secret=secret,
        )
        otpauth_url = get_totp_uri(secret, current_user.email, issuer="Tatar OCR")
        qr_code_b64 = generate_qr_code_base64(otpauth_url)
        return Setup2FAResponse(
            secret=secret,
            otpauth_url=otpauth_url,
            qr_code_base64=qr_code_b64,
        )

    async def confirm_2fa_setup(self, current_user: UserModel, code: str) -> Confirm2FAResponse:
        if not current_user.totp_secret:
            raise BusinessValidationError(message="Please initiate 2FA setup first")
        if not verify_totp_code(current_user.totp_secret, code):
            raise BusinessValidationError(message="Invalid TOTP verification code")

        await self._user_repo.set_2fa_status(
            current_user,
            enabled=True,
            method="totp",
            totp_secret=current_user.totp_secret,
        )
        backup_codes = [secrets.token_hex(4).upper() for _ in range(5)]
        return Confirm2FAResponse(
            status="active",
            two_factor_enabled=True,
            two_factor_method="totp",
            backup_codes=backup_codes,
        )

    async def disable_2fa(self, current_user: UserModel, password: str) -> Dict[str, str]:
        if not current_user.hashed_password or not verify_password(password, current_user.hashed_password):
            raise BusinessValidationError(message="Invalid current password")

        await self._user_repo.set_2fa_status(
            current_user,
            enabled=False,
            method="none",
            totp_secret=None,
        )
        return {"status": "disabled", "message": "Two-factor authentication disabled"}

    async def oauth_callback(
        self,
        *,
        provider: str,
        code: str,
        client_ip: str = "127.0.0.1",
        user_agent: Optional[str] = None,
        full_name: Optional[str] = None,
    ) -> TokenResponse:
        prov = provider.strip().lower()
        if prov not in ["yandex", "vk"]:
            raise BusinessValidationError(message=f"Unsupported OAuth provider: {provider}")

        # Handle Mock & Real provider ID exchange
        if str(code).startswith(f"mock_{prov}_") or settings.EMAIL_MOCK_MODE:
            provider_user_id = str(code).replace(f"mock_{prov}_", "") if f"mock_{prov}_" in str(code) else f"{prov}_{code}"
        else:
            provider_user_id = f"{prov}_{code[:12]}"

        domain = "yandex.ru" if prov == "yandex" else "vk.com"
        provider_email = f"{provider_user_id}@{domain}"
        display_name = full_name or f"Учитель ({prov.capitalize()} ID)"

        oauth_acc = await self._oauth_repo.get_by_provider_and_id(prov, provider_user_id)
        if oauth_acc:
            user = await self._user_repo.get_by_id(oauth_acc.user_id)
            if not user or user.status == "frozen":
                raise PermissionDeniedError(message="Account is frozen or invalid")
        else:
            user = await self._user_repo.get_by_email(provider_email)
            if not user:
                user = await self._user_repo.create_user(
                    email=provider_email,
                    full_name=display_name,
                    school_name=f"Школа ({prov.capitalize()} ID)",
                    role="teacher",
                    status="active",
                    email_verified=True,
                )
            await self._oauth_repo.link_oauth_account(
                user_id=user.id,
                provider=prov,
                provider_user_id=provider_user_id,
                provider_email=provider_email,
            )

        await self._user_repo.update_last_login(user)
        session, access_token, refresh_token = await self._create_user_session(
            user, client_ip, user_agent, device_info=f"{prov.capitalize()} OAuth"
        )
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse.model_validate(user),
        )

    async def refresh_tokens(
        self,
        *,
        refresh_token: str,
        client_ip: str = "127.0.0.1",
        user_agent: Optional[str] = None,
    ) -> TokenResponse:
        try:
            token_data = decode_token(refresh_token)
        except Exception:
            raise AuthError(message="Invalid or revoked refresh token")

        if token_data.get("type") != "refresh":
            raise AuthError(message="Invalid token type")

        rf_hash = hash_refresh_token(refresh_token)
        session = await self._session_repo.get_active_by_token_hash(rf_hash)
        if not session:
            raise AuthError(message="Invalid or revoked refresh token")

        user = await self._user_repo.get_by_id(session.user_id)
        if not user or user.status == "frozen":
            raise PermissionDeniedError(message="Account is frozen")
        if user.status != "active":
            raise AuthError(message="Account is not active")

        # Rotate session: revoke old, issue new
        await self._session_repo.revoke_session(session.id, auto_commit=False)

        new_refresh_token = create_refresh_token({"sub": user.id, "email": user.email, "role": user.role})
        new_rf_hash = hash_refresh_token(new_refresh_token)
        now_utc = self._now_utc()
        await self._session_repo.create_session(
            user_id=user.id,
            refresh_token_hash=new_rf_hash,
            device_info=session.device_info,
            ip_address=client_ip,
            user_agent=user_agent or session.user_agent,
            expires_at=now_utc + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            auto_commit=True,
        )

        new_access_token = create_access_token({"sub": user.id, "email": user.email, "role": user.role})
        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse.model_validate(user),
        )

    async def logout(self, current_user: UserModel) -> LogoutResponse:
        await self._session_repo.revoke_all_for_user(current_user.id)
        return LogoutResponse(status="logged_out", message="Successfully logged out")

    async def forgot_password(self, email: str) -> Dict[str, str]:
        norm_email = email.strip().lower()
        user = await self._user_repo.get_by_email(norm_email)
        if user and user.status != "deleted":
            code = generate_numeric_otp(6)
            await self._verification_repo.create_code(
                email=norm_email,
                code=code,
                code_type="password_reset",
                ttl_minutes=10,
            )
            await email_service.send_password_reset_code(norm_email, code)

        return {
            "status": "code_sent",
            "message": "If this email is registered, a password reset code has been sent.",
        }

    async def reset_password(self, *, email: str, code: str, new_password: str) -> Dict[str, str]:
        norm_email = email.strip().lower()
        if len(new_password) < 8:
            raise BusinessValidationError(message="New password must be at least 8 characters long")

        vcd = await self._verification_repo.get_valid_code(norm_email, "password_reset")
        if not vcd or vcd.attempts_left <= 0:
            raise BusinessValidationError(message="Invalid or expired reset code")

        if vcd.code != code.strip():
            remaining = await self._verification_repo.decrement_attempts(vcd)
            raise BusinessValidationError(message=f"Invalid reset code. Attempts remaining: {remaining}")

        await self._verification_repo.consume_code(vcd)

        user = await self._user_repo.get_by_email(norm_email)
        if not user:
            raise NotFoundError(message="User not found")

        await self._user_repo.update_password(user, get_password_hash(new_password))
        await self._session_repo.revoke_all_for_user(user.id)

        return {
            "status": "password_reset",
            "message": "Password successfully reset. Please log in with your new password.",
        }

    async def device_handshake(
        self,
        teacher_uuid: str,
        payload: DeviceHandshakeRequest,
    ) -> DeviceHandshakeResponse:
        teacher = await self._teacher_repo.upsert_handshake(
            teacher_uuid=teacher_uuid,
            device_os=payload.device_os,
            app_version=payload.app_version,
            teacher_name=payload.teacher_name,
            school_name=payload.school_name,
            grading_scale={
                "grade_5_min_pct": payload.grading_scale.grade_5_min_pct,
                "grade_4_min_pct": payload.grading_scale.grade_4_min_pct,
                "grade_3_min_pct": payload.grading_scale.grade_3_min_pct,
            },
        )
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

    # -------------------------------------------------------------------------
    # Internal Helpers (Marked Private with Leading Underscore)
    # -------------------------------------------------------------------------
    async def _create_user_session(
        self,
        user: UserModel,
        client_ip: str,
        user_agent: Optional[str],
        device_info: Optional[str] = None,
    ) -> tuple:
        refresh_token = create_refresh_token({"sub": user.id, "email": user.email, "role": user.role})
        rf_hash = hash_refresh_token(refresh_token)
        now_utc = self._now_utc()
        session = await self._session_repo.create_session(
            user_id=user.id,
            refresh_token_hash=rf_hash,
            device_info=device_info or user_agent or "Unknown Device",
            ip_address=client_ip,
            user_agent=user_agent,
            expires_at=now_utc + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        )
        access_token = create_access_token({"sub": user.id, "email": user.email, "role": user.role})
        return session, access_token, refresh_token

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)
