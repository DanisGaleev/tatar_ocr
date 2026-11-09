import base64
from datetime import datetime, timedelta, timezone
import hashlib
import io
import secrets
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from passlib.context import CryptContext
import pyotp
import qrcode
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.school import ClassModel
from app.models.teacher import TeacherModel
from app.models.user import UserModel

from slowapi import Limiter
from slowapi.util import get_remote_address

pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    auto_error=False,
)
limiter = Limiter(key_func=get_remote_address, enabled=settings.RATE_LIMIT_ENABLED)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against hashed password."""
    if not hashed_password:
        return False
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate Argon2id/bcrypt password hash."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create signed JWT access token."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({
        "exp": expire,
        "iat": now,
        "type": to_encode.get("type", "access"),
        "jti": secrets.token_hex(16),
    })
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create signed JWT refresh token."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({
        "exp": expire,
        "iat": now,
        "type": "refresh",
        "jti": secrets.token_hex(16),
    })
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt


def decode_token(token: str) -> dict:
    """Decode and validate JWT token claims."""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (jwt.PyJWTError, Exception):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def hash_refresh_token(token: str) -> str:
    """Return SHA-256 hash of refresh token for session storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_totp_secret() -> str:
    """Generate Base32 secret for Google Authenticator / 2FA."""
    return pyotp.random_base32()


def get_totp_uri(secret: str, email: str, issuer: Optional[str] = None) -> str:
    """Generate otpauth:// URI for authenticator apps."""
    totp = pyotp.TOTP(secret)
    issuer_name = issuer or "Кара куян (Tatar OCR)"
    return totp.provisioning_uri(name=email, issuer_name=issuer_name)


def generate_qr_code_base64(otpauth_url: str) -> str:
    """Generate Base64 encoded PNG QR code image from otpauth URL."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=6,
        border=3,
    )
    qr.add_data(otpauth_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def verify_totp_code(secret: str, code: str) -> bool:
    """Verify 6-digit TOTP code against secret with 1-step window."""
    if not secret or not code:
        return False
    try:
        totp = pyotp.TOTP(secret)
        return bool(totp.verify(str(code).strip(), valid_window=1))
    except Exception:
        return False


def generate_numeric_otp(length: int = 6) -> str:
    """Generate cryptographically secure N-digit numeric OTP code."""
    digits = [str(secrets.randbelow(10)) for _ in range(length)]
    return "".join(digits)


# Re-export core dependencies for backward compatibility
from app.core.dependencies import (  # noqa: E402
    get_current_active_user,
    get_current_admin_user,
    require_admin,
    require_superadmin,
)


async def get_current_active_user_strict(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> UserModel:
    """Strictly requires Bearer token, active user status."""
    if not token:
        from app.core.exceptions import AuthError
        raise AuthError(message="Authentication required")
    return await get_current_active_user(request=request, token=token, db=db)

