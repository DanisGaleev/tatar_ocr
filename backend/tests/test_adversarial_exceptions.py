import asyncio
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
import pytest
from pydantic import BaseModel, Field
from httpx import AsyncClient, ASGITransport
from fastapi import APIRouter, Depends, FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi.errors import RateLimitExceeded

from app.main import app
from app.core.exceptions import (
    AppException,
    AuthError,
    PermissionDeniedError,
    NotFoundError,
    BusinessValidationError,
    RateLimitError,
    ConflictError,
)

# Adversarial router to test every exception path and edge case
adversarial_router = APIRouter(prefix="/adversarial-errors", tags=["AdversarialErrors"])


class NestedItem(BaseModel):
    item_id: uuid.UUID
    timestamp: datetime
    price: Decimal
    tags: List[str] = Field(default_factory=list)


class ComplexDetailPayload(BaseModel):
    user_id: uuid.UUID
    created_at: datetime
    active: bool
    nested: NestedItem
    extra: Dict[str, Any] = Field(default_factory=dict)


@adversarial_router.get("/subclass/{error_type}")
async def trigger_subclass(error_type: str):
    if error_type == "auth":
        raise AuthError(message="Танылу хатасы (Auth failed)", details={"scope": "admin"})
    elif error_type == "permission":
        raise PermissionDeniedError(message="Рөхсәт юк (Forbidden)", details={"required_role": "admin"})
    elif error_type == "not_found":
        raise NotFoundError(message="Сыйныф табылмады (Class not found)", details={"id": "cls_999"})
    elif error_type == "validation":
        raise BusinessValidationError(message="Мәгълүмат дөрес түгел", details=[{"loc": ["email"], "msg": "invalid"}])
    elif error_type == "validation_422":
        raise BusinessValidationError(message="Unprocessable Entity", status_code=422, code="UNPROCESSABLE")
    elif error_type == "rate_limit":
        raise RateLimitError(message="Бик күп сораулар", retry_after=45, details={"limit": "10/min"})
    elif error_type == "conflict":
        raise ConflictError(message="Мондый кулланучы бар инде", details={"field": "email"})
    elif error_type == "app_base":
        raise AppException(message="База хатасы", status_code=500, code="BASE_ERROR", details={"origin": "core"})
    elif error_type == "custom_status":
        raise AppException(message="I'm a teapot", status_code=418, code="TEAPOT", details="custom")
    else:
        raise AppException(message="Unknown type")


@adversarial_router.get("/complex-serialization")
async def trigger_complex_serialization():
    payload = ComplexDetailPayload(
        user_id=uuid.UUID("a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"),
        created_at=datetime(2026, 11, 7, 21, 30, 0, tzinfo=timezone.utc),
        active=True,
        nested=NestedItem(
            item_id=uuid.UUID("11111111-2222-3333-4444-555555555555"),
            timestamp=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            price=Decimal("149.99"),
            tags=["tatar", "ocr", "ai"],
        ),
        extra={
            "set_of_ids": {10, 20, 30},
            "sub_uuid": uuid.UUID("99999999-8888-7777-6666-555555555555"),
            "naive_dt": datetime(2026, 6, 15, 12, 0, 0),
        },
    )
    raise BusinessValidationError(
        message="Complex serialization test",
        details=payload,
    )


@adversarial_router.get("/edge-cases/{case_name}")
async def trigger_edge_cases(case_name: str):
    if case_name == "empty_message":
        raise AppException(message="", status_code=400, code="EMPTY_MSG")
    elif case_name == "null_details":
        raise AppException(message="Null details test", status_code=400, details=None)
    elif case_name == "empty_dict_details":
        raise AppException(message="Empty dict test", status_code=400, details={})
    elif case_name == "empty_list_details":
        raise AppException(message="Empty list test", status_code=400, details=[])
    elif case_name == "bool_false_details":
        raise AppException(message="Bool false test", status_code=400, details=False)
    elif case_name == "zero_int_details":
        raise AppException(message="Zero int test", status_code=400, details=0)
    elif case_name == "unicode_extremes":
        raise AppException(
            message="Күнегүләр: Ә, Ө, Ү, Җ, Ң, Һ (Unicode letters) \u2603 \U0001F680 '\"`\n\t",
            status_code=400,
            code="UNICODE_EXTREME",
            details={"chars": "ӘӨҮҖҢҺ", "emoji": "🚀", "newline": "line1\nline2"},
        )


async def failing_dependency():
    raise AuthError(message="Dependency auth failure", details={"step": "jwt_decode"})


@adversarial_router.get("/in-dependency")
async def trigger_in_dependency(dep=Depends(failing_dependency)):
    return {"status": "should_not_reach_here"}


@adversarial_router.get("/starlette-http-error/{status_code}")
async def trigger_starlette_http(status_code: int):
    if status_code == 404:
        raise StarletteHTTPException(status_code=404, detail="Starlette not found")
    elif status_code == 403:
        raise StarletteHTTPException(status_code=403, detail="Starlette forbidden")
    elif status_code == 401:
        raise StarletteHTTPException(
            status_code=401,
            detail="Starlette unauthorized",
            headers={"WWW-Authenticate": "Bearer error=\"invalid_token\""},
        )
    else:
        raise StarletteHTTPException(status_code=status_code, detail="Starlette error")


@adversarial_router.get("/generic-crash")
async def trigger_generic_crash():
    # Intentionally trigger unhandled ZeroDivisionError
    _ = 1 / 0


@adversarial_router.get("/slowapi-rate-limit")
async def trigger_slowapi_rate_limit():
    mock_limit = type("MockLimit", (), {"error_message": "100/minute", "limit": "100/minute"})()
    raise RateLimitExceeded(mock_limit)


# Mount adversarial router once into the app
app.include_router(adversarial_router)


# ==============================================================================
# TEST SUITE
# ==============================================================================

class TestAdversarialExceptionHierarchy:
    """Tests all AppException subclasses and contractual attributes."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "err_key,expected_status,expected_code,expected_detail,expected_header",
        [
            ("auth", 401, "AUTH_ERROR", "Танылу хатасы (Auth failed)", ("www-authenticate", "Bearer")),
            ("permission", 403, "PERMISSION_DENIED", "Рөхсәт юк (Forbidden)", None),
            ("not_found", 404, "NOT_FOUND", "Сыйныф табылмады (Class not found)", None),
            ("validation", 400, "BUSINESS_VALIDATION_ERROR", "Мәгълүмат дөрес түгел", None),
            ("validation_422", 422, "UNPROCESSABLE", "Unprocessable Entity", None),
            ("rate_limit", 429, "RATE_LIMIT_EXCEEDED", "Бик күп сораулар", ("retry-after", "45")),
            ("conflict", 409, "CONFLICT", "Мондый кулланучы бар инде", None),
            ("app_base", 500, "BASE_ERROR", "База хатасы", None),
            ("custom_status", 418, "TEAPOT", "I'm a teapot", None),
        ],
    )
    async def test_subclass_contract(
        self, err_key, expected_status, expected_code, expected_detail, expected_header
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get(f"/adversarial-errors/subclass/{err_key}")
            assert res.status_code == expected_status
            data = res.json()

            # Contractual fields must all be present
            assert "detail" in data
            assert "status_code" in data
            assert "error_code" in data
            assert "details" in data

            assert data["status_code"] == expected_status
            assert data["error_code"] == expected_code
            assert data["detail"] == expected_detail
            if err_key == "validation_422":
                assert data["details"] is None
            else:
                assert data["details"] is not None

            if expected_header:
                h_name, h_val = expected_header
                assert res.headers.get(h_name) == h_val


class TestComplexSerialization:
    """Stress tests serialization of Pydantic models, datetimes, UUIDs, sets, Decimals."""

    @pytest.mark.asyncio
    async def test_pydantic_datetime_uuid_nested(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/adversarial-errors/complex-serialization")
            assert res.status_code == 400
            data = res.json()

            assert data["detail"] == "Complex serialization test"
            assert data["status_code"] == 400
            assert data["error_code"] == "BUSINESS_VALIDATION_ERROR"

            details = data["details"]
            assert details["user_id"] == "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"
            assert "2026-11-07T21:30:00" in details["created_at"]
            assert details["active"] is True

            nested = details["nested"]
            assert nested["item_id"] == "11111111-2222-3333-4444-555555555555"
            assert nested["price"] == "149.99"
            assert nested["tags"] == ["tatar", "ocr", "ai"]

            extra = details["extra"]
            # Set must be converted to list
            assert sorted(extra["set_of_ids"]) == [10, 20, 30]
            assert extra["sub_uuid"] == "99999999-8888-7777-6666-555555555555"
            assert "2026-06-15" in extra["naive_dt"]


class TestAdversarialEdgeCases:
    """Tests extreme cases: empty strings, unicode letters, emojis, falsey details."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "case_name,expected_detail,expected_details",
        [
            ("empty_message", "", None),
            ("null_details", "Null details test", None),
            ("empty_dict_details", "Empty dict test", {}),
            ("empty_list_details", "Empty list test", []),
            ("bool_false_details", "Bool false test", False),
            ("zero_int_details", "Zero int test", 0),
        ],
    )
    async def test_edge_cases(self, case_name, expected_detail, expected_details):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get(f"/adversarial-errors/edge-cases/{case_name}")
            assert res.status_code == 400
            data = res.json()
            assert data["detail"] == expected_detail
            assert data["details"] == expected_details

    @pytest.mark.asyncio
    async def test_unicode_and_emojis(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/adversarial-errors/edge-cases/unicode_extremes")
            assert res.status_code == 400
            data = res.json()
            assert "Ә, Ө, Ү, Җ, Ң, Һ" in data["detail"]
            assert "🚀" in data["detail"]
            assert data["details"]["chars"] == "ӘӨҮҖҢҺ"
            assert data["details"]["emoji"] == "🚀"
            assert data["details"]["newline"] == "line1\nline2"


class TestFastAPIDependencyExceptions:
    """Ensures exceptions raised in FastAPI Depends() are intercepted cleanly."""

    @pytest.mark.asyncio
    async def test_exception_in_dependency(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/adversarial-errors/in-dependency")
            assert res.status_code == 401
            data = res.json()
            assert data["detail"] == "Dependency auth failure"
            assert data["status_code"] == 401
            assert data["error_code"] == "AUTH_ERROR"
            assert data["details"] == {"step": "jwt_decode"}
            assert res.headers.get("www-authenticate") == "Bearer"


class TestStarletteAndGenericExceptions:
    """Tests Starlette HTTPException and generic Exception handlers."""

    @pytest.mark.asyncio
    async def test_starlette_http_not_found(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/adversarial-errors/starlette-http-error/404")
            assert res.status_code == 404
            data = res.json()
            assert data["detail"] == "Starlette not found"
            assert data["status_code"] == 404

    @pytest.mark.asyncio
    async def test_starlette_http_forbidden(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/adversarial-errors/starlette-http-error/403")
            assert res.status_code == 403
            data = res.json()
            assert data["detail"] == "Starlette forbidden"
            assert data["status_code"] == 403

    @pytest.mark.asyncio
    async def test_unhandled_generic_crash_no_traceback_leak(self):
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
        ) as client:
            res = await client.get("/adversarial-errors/generic-crash")
            assert res.status_code == 500
            data = res.json()
            # Must return clean generic message without leaking ZeroDivisionError or traceback
            assert data["detail"] == "Internal server error"
            assert data["status_code"] == 500
            assert "ZeroDivisionError" not in str(data)
            assert "Traceback" not in str(data)

    @pytest.mark.asyncio
    async def test_slowapi_rate_limit_exceeded(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/adversarial-errors/slowapi-rate-limit")
            assert res.status_code == 429
            data = res.json()
            assert "Rate limit exceeded" in data["detail"]
            assert data["status_code"] == 429
            assert data["error_code"] == "RATE_LIMIT_EXCEEDED"


class TestHighConcurrencyBurst:
    """Stress tests high concurrency / burst requests triggering exceptions."""

    @pytest.mark.asyncio
    async def test_concurrency_burst_100_requests(self):
        endpoints = [
            ("/adversarial-errors/subclass/auth", 401, "AUTH_ERROR"),
            ("/adversarial-errors/subclass/permission", 403, "PERMISSION_DENIED"),
            ("/adversarial-errors/subclass/not_found", 404, "NOT_FOUND"),
            ("/adversarial-errors/subclass/validation", 400, "BUSINESS_VALIDATION_ERROR"),
            ("/adversarial-errors/subclass/rate_limit", 429, "RATE_LIMIT_EXCEEDED"),
            ("/adversarial-errors/subclass/conflict", 409, "CONFLICT"),
            ("/adversarial-errors/complex-serialization", 400, "BUSINESS_VALIDATION_ERROR"),
            ("/adversarial-errors/starlette-http-error/404", 404, None),
            ("/adversarial-errors/generic-crash", 500, None),
            ("/adversarial-errors/in-dependency", 401, "AUTH_ERROR"),
        ]

        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
        ) as client:
            # 10 rounds of 10 endpoints = 100 concurrent requests
            tasks = []
            for _ in range(10):
                for path, status_code, err_code in endpoints:
                    tasks.append((client.get(path), status_code, err_code))

            results = await asyncio.gather(*(t[0] for t in tasks))

            assert len(results) == 100
            for i, res in enumerate(results):
                expected_status = tasks[i][1]
                expected_code = tasks[i][2]

                assert res.status_code == expected_status, f"Request {i} expected {expected_status} but got {res.status_code}"
                data = res.json()
                assert "detail" in data
                assert data["status_code"] == expected_status

                if expected_code is not None:
                    assert data["error_code"] == expected_code, f"Request {i} error_code mismatch: {data.get('error_code')} != {expected_code}"
