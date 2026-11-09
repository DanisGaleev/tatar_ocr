from typing import Any, Dict, Optional


class AppException(Exception):
    """Base exception for all application-level errors."""

    def __init__(
        self,
        message: str = "An internal application error occurred",
        status_code: int = 500,
        code: str = "INTERNAL_SERVER_ERROR",
        details: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details
        self.headers = headers

    @property
    def error_code(self) -> str:
        """Alias for code attribute for compatibility."""
        return self.code

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary response payload."""
        return {
            "detail": self.message,
            "status_code": self.status_code,
            "error_code": self.code,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} status_code={self.status_code} "
            f"code={self.code} message={self.message!r}>"
        )


class AuthError(AppException):
    """Raised when authentication credentials are missing, invalid, or expired (HTTP 401)."""

    def __init__(
        self,
        message: str = "Authentication failed",
        status_code: int = 401,
        code: str = "AUTH_ERROR",
        details: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        default_headers = {"WWW-Authenticate": "Bearer"}
        if headers:
            default_headers.update(headers)
        super().__init__(
            message=message,
            status_code=status_code,
            code=code,
            details=details,
            headers=default_headers,
        )


class PermissionDeniedError(AppException):
    """Raised when user lacks permission, IDOR boundary is violated, or account is frozen (HTTP 403)."""

    def __init__(
        self,
        message: str = "Permission denied",
        status_code: int = 403,
        code: str = "PERMISSION_DENIED",
        details: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            code=code,
            details=details,
            headers=headers,
        )


class NotFoundError(AppException):
    """Raised when a requested resource does not exist (HTTP 404)."""

    def __init__(
        self,
        message: str = "Resource not found",
        status_code: int = 404,
        code: str = "NOT_FOUND",
        details: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            code=code,
            details=details,
            headers=headers,
        )


class BusinessValidationError(AppException):
    """Raised when business logic validation fails (HTTP 400 default, configurable e.g. 422)."""

    def __init__(
        self,
        message: str = "Validation error",
        status_code: int = 400,
        code: str = "BUSINESS_VALIDATION_ERROR",
        details: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            code=code,
            details=details,
            headers=headers,
        )


class RateLimitError(AppException):
    """Raised when rate limits or account lockout thresholds are exceeded (HTTP 429)."""

    def __init__(
        self,
        message: str = "Rate limit exceeded. Please try again later.",
        status_code: int = 429,
        code: str = "RATE_LIMIT_EXCEEDED",
        details: Optional[Any] = None,
        retry_after: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        req_headers = dict(headers) if headers else {}
        if retry_after is not None:
            req_headers["Retry-After"] = str(retry_after)
        super().__init__(
            message=message,
            status_code=status_code,
            code=code,
            details=details,
            headers=req_headers if req_headers else None,
        )


class ConflictError(AppException):
    """Raised on resource conflict or unique constraint violation (HTTP 409)."""

    def __init__(
        self,
        message: str = "Resource conflict",
        status_code: int = 409,
        code: str = "CONFLICT",
        details: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            code=code,
            details=details,
            headers=headers,
        )


__all__ = [
    "AppException",
    "AuthError",
    "PermissionDeniedError",
    "NotFoundError",
    "BusinessValidationError",
    "RateLimitError",
    "ConflictError",
]
