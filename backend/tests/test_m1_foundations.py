import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from httpx import AsyncClient, ASGITransport
from fastapi import APIRouter
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
from slowapi.errors import RateLimitExceeded

from app.main import app
from app.core.database import Base, get_db
from app.core.exceptions import (
    AppException,
    AuthError,
    PermissionDeniedError,
    NotFoundError,
    BusinessValidationError,
    RateLimitError,
    ConflictError,
)
from app.models.user import UserModel
from app.models.session import UserSessionModel
from app.models.verification import VerificationCodeModel
from app.models.school import ClassModel, StudentModel, ClassAssignmentModel
from app.models.submission import SubmissionModel
from app.models.test import AssembledTestModel
from app.models.audit import AuditLogModel
from app.models.oauth import OAuthAccountModel
from app.models.teacher import TeacherModel

from app.repositories import (
    BaseRepository,
    UserRepository,
    SessionRepository,
    VerificationCodeRepository,
    ClassRepository,
    StudentRepository,
    SubmissionRepository,
    AssignmentRepository,
    AuditRepository,
    OAuthRepository,
    TeacherRepository,
)

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

m1_engine = create_async_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
M1SessionLocal = async_sessionmaker(
    bind=m1_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


@pytest_asyncio.fixture(scope="module", autouse=True)
async def init_test_db():
    async with m1_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with m1_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session():
    async with M1SessionLocal() as session:
        yield session
        await session.rollback()


# Router to test HTTP exception handling through FastAPI pipeline
error_test_router = APIRouter(prefix="/test-errors", tags=["TestErrors"])


@error_test_router.get("/auth-error")
async def trigger_auth_error():
    raise AuthError(message="Invalid token", details={"reason": "expired"})


@error_test_router.get("/permission-denied")
async def trigger_permission_denied():
    raise PermissionDeniedError(message="Forbidden access")


@error_test_router.get("/not-found")
async def trigger_not_found():
    raise NotFoundError(message="Class not found")


@error_test_router.get("/validation-error")
async def trigger_validation_error():
    raise BusinessValidationError(message="Invalid email syntax", status_code=422)


@error_test_router.get("/rate-limit")
async def trigger_rate_limit():
    raise RateLimitError(message="Too many requests", retry_after=120)


@error_test_router.get("/conflict")
async def trigger_conflict():
    raise ConflictError(message="Account with this email already exists")


@error_test_router.get("/rate-limit-exceeded")
async def trigger_slowapi_rate_limit():
    mock_limit = type("MockLimit", (), {"error_message": "5 per minute", "limit": "5/minute"})()
    raise RateLimitExceeded(mock_limit)


app.include_router(error_test_router)


# --------------------------------------------------------------------------
# 1. Exception Hierarchy Unit Tests
# --------------------------------------------------------------------------
class TestExceptionsHierarchy:
    def test_app_exception_defaults(self):
        exc = AppException()
        assert exc.status_code == 500
        assert exc.code == "INTERNAL_SERVER_ERROR"
        assert exc.error_code == "INTERNAL_SERVER_ERROR"
        assert exc.message == "An internal application error occurred"
        assert exc.details is None
        assert "500" in repr(exc)

        payload = exc.to_dict()
        assert payload["detail"] == exc.message
        assert payload["status_code"] == 500
        assert payload["error_code"] == "INTERNAL_SERVER_ERROR"

    def test_auth_error(self):
        exc = AuthError(message="Bad credentials", details={"user": "test@test.com"})
        assert exc.status_code == 401
        assert exc.code == "AUTH_ERROR"
        assert exc.error_code == "AUTH_ERROR"
        assert exc.headers == {"WWW-Authenticate": "Bearer"}
        assert exc.details == {"user": "test@test.com"}

    def test_permission_denied_error(self):
        exc = PermissionDeniedError(message="Access denied")
        assert exc.status_code == 403
        assert exc.code == "PERMISSION_DENIED"

    def test_not_found_error(self):
        exc = NotFoundError(message="User not found")
        assert exc.status_code == 404
        assert exc.code == "NOT_FOUND"

    def test_business_validation_error(self):
        exc400 = BusinessValidationError("Invalid parameter")
        assert exc400.status_code == 400
        assert exc400.code == "BUSINESS_VALIDATION_ERROR"

        exc422 = BusinessValidationError("Malformed field", status_code=422)
        assert exc422.status_code == 422

    def test_rate_limit_error(self):
        exc = RateLimitError(retry_after=60)
        assert exc.status_code == 429
        assert exc.code == "RATE_LIMIT_EXCEEDED"
        assert exc.headers == {"Retry-After": "60"}

    def test_conflict_error(self):
        exc = ConflictError(message="Duplicate resource")
        assert exc.status_code == 409
        assert exc.code == "CONFLICT"


# --------------------------------------------------------------------------
# 2. HTTP Exception Handler Integration Tests
# --------------------------------------------------------------------------
class TestHTTPExceptionHandling:
    @pytest.mark.asyncio
    async def test_auth_error_http(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/test-errors/auth-error")
            assert res.status_code == 401
            data = res.json()
            assert "detail" in data
            assert data["detail"] == "Invalid token"
            assert data["status_code"] == 401
            assert data["error_code"] == "AUTH_ERROR"
            assert data["details"] == {"reason": "expired"}
            assert res.headers.get("www-authenticate") == "Bearer"

    @pytest.mark.asyncio
    async def test_permission_denied_http(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/test-errors/permission-denied")
            assert res.status_code == 403
            data = res.json()
            assert data["detail"] == "Forbidden access"
            assert data["error_code"] == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_not_found_http(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/test-errors/not-found")
            assert res.status_code == 404
            data = res.json()
            assert data["detail"] == "Class not found"
            assert data["error_code"] == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_validation_error_http(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/test-errors/validation-error")
            assert res.status_code == 422
            data = res.json()
            assert data["detail"] == "Invalid email syntax"
            assert data["status_code"] == 422
            assert data["error_code"] == "BUSINESS_VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_rate_limit_http(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/test-errors/rate-limit")
            assert res.status_code == 429
            data = res.json()
            assert data["detail"] == "Too many requests"
            assert data["error_code"] == "RATE_LIMIT_EXCEEDED"
            assert res.headers.get("retry-after") == "120"

    @pytest.mark.asyncio
    async def test_conflict_http(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/test-errors/conflict")
            assert res.status_code == 409
            data = res.json()
            assert data["detail"] == "Account with this email already exists"
            assert data["error_code"] == "CONFLICT"

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_slowapi_http(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/test-errors/rate-limit-exceeded")
            assert res.status_code == 429
            data = res.json()
            assert "detail" in data
            assert "Rate limit exceeded" in data["detail"]
            assert data["error_code"] == "RATE_LIMIT_EXCEEDED"


# --------------------------------------------------------------------------
# 3. BaseRepository Generic Tests
# --------------------------------------------------------------------------
class TestBaseRepository:
    @pytest.mark.asyncio
    async def test_base_crud(self, db_session: AsyncSession):
        repo = BaseRepository[ClassModel](db_session, ClassModel)
        # Create
        cls = await repo.create({
            "class_id": "cls_base_test",
            "name": "Base Test Class",
            "subject": "tatar_language",
            "academic_year": "2026-2027",
        })
        assert cls.class_id == "cls_base_test"

        # Get & Get_by_id
        fetched = await repo.get("cls_base_test")
        assert fetched is not None
        assert fetched.name == "Base Test Class"
        assert await repo.get_by_id("cls_base_test") is not None

        # Count & Get_all / Get_multi
        count = await repo.count()
        assert count >= 1
        all_classes = await repo.get_all(skip=0, limit=10)
        assert len(all_classes) >= 1
        multi_classes = await repo.get_multi(skip=0, limit=10)
        assert len(multi_classes) >= 1

        # Update
        updated = await repo.update(cls, {"name": "Updated Base Class"})
        assert updated.name == "Updated Base Class"

        # Delete by ID
        del_success = await repo.delete("cls_base_test")
        assert del_success is True
        assert await repo.get("cls_base_test") is None


# --------------------------------------------------------------------------
# 4. UserRepository Tests
# --------------------------------------------------------------------------
class TestUserRepository:
    @pytest.mark.asyncio
    async def test_user_lifecycle(self, db_session: AsyncSession):
        repo = UserRepository(db_session)

        # Create
        user = await repo.create_user(
            email="teacher1@example.com",
            hashed_password="hashed_pw_secret",
            full_name="Каримов Азат Равилевич",
            school_name="Гимназия №2",
            role="teacher",
            status="unverified",
        )
        assert user.id.startswith("usr_")
        assert user.email == "teacher1@example.com"
        assert user.failed_login_attempts == 0
        assert user.locked_until is None

        # Lookup by ID and Email (case-insensitive)
        assert await repo.get_by_id(user.id) is not None
        by_email = await repo.get_by_email("TEACHER1@example.COM")
        assert by_email is not None
        assert by_email.id == user.id

        # Update user
        await repo.update_user(user, {"school_name": "Гимназия №7"})
        assert user.school_name == "Гимназия №7"

        # Email verification
        await repo.mark_email_verified(user)
        assert user.email_verified is True
        assert user.status == "active"

        # 2FA configuration
        await repo.set_2fa_status(user, enabled=True, method="totp", totp_secret="BASE32SECRET")
        assert user.two_factor_enabled is True
        assert user.totp_secret == "BASE32SECRET"

        # Last login update
        await repo.update_last_login(user)
        assert user.last_login_at is not None

        # Brute force lockout cycle
        for _ in range(4):
            await repo.increment_failed_attempts(user, max_attempts=5, lockout_minutes=15)
        assert user.failed_login_attempts == 4
        assert user.locked_until is None

        # 5th failed attempt locks the account
        await repo.increment_failed_attempts(user, max_attempts=5, lockout_minutes=15)
        assert user.failed_login_attempts == 5
        assert user.locked_until is not None
        locked = (
            user.locked_until
            if user.locked_until.tzinfo
            else user.locked_until.replace(tzinfo=timezone.utc)
        )
        assert locked > datetime.now(timezone.utc)

        # Reset failed attempts unlocks
        await repo.reset_failed_attempts(user)
        assert user.failed_login_attempts == 0
        assert user.locked_until is None

        # Password update also clears lockout
        user.failed_login_attempts = 3
        await repo.update_password(user, "new_hashed_secret")
        assert user.failed_login_attempts == 0
        assert user.hashed_password == "new_hashed_secret"

        # Status and soft-delete
        await repo.set_status(user, "frozen")
        assert user.status == "frozen"
        await repo.soft_delete(user)
        assert user.status == "deleted"

    @pytest.mark.asyncio
    async def test_user_list_and_search(self, db_session: AsyncSession):
        repo = UserRepository(db_session)
        # Create users for filtering
        await repo.create_user(
            email="filter_admin@school.com",
            full_name="Администратор Системы",
            role="admin",
            status="active",
        )
        await repo.create_user(
            email="filter_teacher@school.com",
            full_name="Мөгаллим Ирек",
            role="teacher",
            status="frozen",
            school_name="Татар Мәктәбе",
        )

        # List by role
        admins, total_admins = await repo.list_users(role="admin")
        assert any(u.email == "filter_admin@school.com" for u in admins)
        assert total_admins >= 1

        # List by status
        frozen, total_frozen = await repo.list_users(status="frozen")
        assert any(u.email == "filter_teacher@school.com" for u in frozen)

        # Search by full name substring
        searched, total_s = await repo.list_users(search="Ирек")
        assert any(u.email == "filter_teacher@school.com" for u in searched)

        # Search by school name substring
        searched_school, _ = await repo.list_users(search="Мәктәбе")
        assert any(u.email == "filter_teacher@school.com" for u in searched_school)


# --------------------------------------------------------------------------
# 5. SessionRepository Tests
# --------------------------------------------------------------------------
class TestSessionRepository:
    @pytest.mark.asyncio
    async def test_session_lifecycle(self, db_session: AsyncSession):
        u_repo = UserRepository(db_session)
        user = await u_repo.create_user(
            email="session_user@test.com",
            full_name="Студент Тест",
        )

        s_repo = SessionRepository(db_session)
        future_exp = datetime.now(timezone.utc) + timedelta(days=7)

        # Create session
        sess = await s_repo.create_session(
            user_id=user.id,
            refresh_token_hash="sha256_mock_hash_123",
            expires_at=future_exp,
            device_info="Pixel 8",
            ip_address="192.168.1.1",
            user_agent="TatarOCR/1.0",
        )
        assert sess.id.startswith("ses_")
        assert sess.revoked_at is None

        # Lookups
        by_hash = await s_repo.get_by_token_hash("sha256_mock_hash_123")
        assert by_hash is not None
        assert by_hash.id == sess.id

        active_by_hash = await s_repo.get_active_by_token_hash("sha256_mock_hash_123")
        assert active_by_hash is not None

        # List by user
        user_sessions = await s_repo.list_by_user(user.id, active_only=True)
        assert len(user_sessions) == 1

        # List all active
        active_list, total_act = await s_repo.list_all_active()
        assert any(s.id == sess.id for s in active_list)
        assert total_act >= 1

        # Revoke single session
        revoked = await s_repo.revoke_session(sess.id)
        assert revoked is True
        assert (await s_repo.get_active_by_token_hash("sha256_mock_hash_123")) is None

        # Create another session and revoke all for user
        sess2 = await s_repo.create_session(
            user_id=user.id,
            refresh_token_hash="sha256_mock_hash_456",
            expires_at=future_exp,
        )
        revoked_count = await s_repo.revoke_all_for_user(user.id)
        assert revoked_count >= 1
        assert (await s_repo.get_active_by_token_hash("sha256_mock_hash_456")) is None


# --------------------------------------------------------------------------
# 6. VerificationCodeRepository Tests
# --------------------------------------------------------------------------
class TestVerificationCodeRepository:
    @pytest.mark.asyncio
    async def test_verification_lifecycle(self, db_session: AsyncSession):
        v_repo = VerificationCodeRepository(db_session)
        email = "verify@test.com"
        code_type = "register_verify"

        # Create OTP
        v_code = await v_repo.create_code(
            email=email,
            code="123456",
            code_type=code_type,
            ttl_minutes=10,
            attempts_left=5,
        )
        assert v_code.id.startswith("vcd_")
        assert v_code.attempts_left == 5

        # Get valid code
        valid = await v_repo.get_valid_code(email, code_type)
        assert valid is not None
        assert valid.code == "123456"

        # Decrement attempts
        attempts = await v_repo.decrement_attempts(valid)
        assert attempts == 4

        # Consume code
        consumed = await v_repo.consume_code(valid)
        assert consumed.is_used is True

        # Now get_valid_code returns None since it was consumed
        assert await v_repo.get_valid_code(email, code_type) is None

        # Cooldown check
        last_created = await v_repo.get_last_created_at(email, code_type)
        assert last_created is not None

        # Invalidate all
        await v_repo.create_code(email=email, code="654321", code_type=code_type)
        inv_count = await v_repo.invalidate_all(email, code_type)
        assert inv_count >= 1
        assert await v_repo.get_valid_code(email, code_type) is None


# --------------------------------------------------------------------------
# 7. ClassRepository & Cascade Tests
# --------------------------------------------------------------------------
class TestClassRepository:
    @pytest.mark.asyncio
    async def test_class_cascade_and_counts(self, db_session: AsyncSession):
        c_repo = ClassRepository(db_session)
        s_repo = StudentRepository(db_session)
        sub_repo = SubmissionRepository(db_session)

        # Create class
        cls = await c_repo.create_class(
            class_id="cls_cascade_test",
            name="8-Б",
            teacher_uuid="teacher_uuid_xyz",
        )
        assert cls.class_id == "cls_cascade_test"

        # Add students to class
        stu = await s_repo.create_student(
            class_id="cls_cascade_test",
            last_name="Хәкимов",
            first_name="Булат",
        )

        # Add submission
        await sub_repo.create_submission({
            "assignment_id": "TAT-08-01",
            "class_id": "cls_cascade_test",
            "student_id": stu.student_id,
            "student_name": "Хәкимов Б.",
            "overall_score": 7.0,
            "max_score": 8.0,
            "final_grade": 5,
        })

        # List classes with student counts
        counts = await c_repo.list_classes_with_student_counts(teacher_uuid="teacher_uuid_xyz")
        found = next((item for item in counts if item[0].class_id == "cls_cascade_test"), None)
        assert found is not None
        assert found[1] == 1  # 1 student

        # Cascade delete
        del_res = await c_repo.delete_cascade("cls_cascade_test")
        assert del_res is True

        # Verify class, student, and submission were all deleted
        assert await c_repo.get_by_id("cls_cascade_test") is None
        assert await s_repo.get_by_id(stu.student_id) is None
        subs = await sub_repo.list_by_class("cls_cascade_test")
        assert len(subs) == 0


# --------------------------------------------------------------------------
# 8. StudentRepository Tests
# --------------------------------------------------------------------------
class TestStudentRepository:
    @pytest.mark.asyncio
    async def test_student_roster(self, db_session: AsyncSession):
        c_repo = ClassRepository(db_session)
        await c_repo.get_or_create_class("cls_student_roster", name="9-А")

        s_repo = StudentRepository(db_session)
        # Create single with automated formatting
        s1 = await s_repo.create_student(
            class_id="cls_student_roster",
            last_name="Нәҗмиев",
            first_name="Илдар",
            middle_name="Рамилевич",
        )
        assert s1.full_name == "Нәҗмиев Илдар Рамилевич"
        assert s1.short_name == "Нәҗмиев И."

        # Bulk create
        bulk_res = await s_repo.bulk_create([
            {"class_id": "cls_student_roster", "last_name": "Ахматов", "first_name": "Дамир"},
            {"class_id": "cls_student_roster", "last_name": "Сабирова", "first_name": "Алия"},
        ])
        assert len(bulk_res) == 2

        # Count and roster listing
        assert await s_repo.count_by_class("cls_student_roster") == 3
        roster = await s_repo.list_by_class("cls_student_roster")
        assert len(roster) == 3
        # Ordered by last_name
        assert roster[0].last_name == "Ахматов"

        # Update student
        await s_repo.update_student(s1, last_name="Нәҗмиев", first_name="Ильдар")
        assert s1.full_name == "Нәҗмиев Ильдар Рамилевич"

        # Soft delete
        await s_repo.soft_delete(s1.student_id)
        assert s1.is_active is False
        assert await s_repo.count_by_class("cls_student_roster", active_only=True) == 2

        # Clear class students
        cleared = await s_repo.clear_class_students("cls_student_roster")
        assert cleared >= 2
        assert await s_repo.count_by_class("cls_student_roster", active_only=False) == 0


# --------------------------------------------------------------------------
# 9. SubmissionRepository Tests
# --------------------------------------------------------------------------
class TestSubmissionRepository:
    @pytest.mark.asyncio
    async def test_submission_upsert_and_queries(self, db_session: AsyncSession):
        repo = SubmissionRepository(db_session)
        client_uuid = "client-uuid-sync-test-999"

        payload = {
            "client_submission_uuid": client_uuid,
            "assignment_id": "TAT-07-01",
            "class_id": "cls_sub_test",
            "student_id": "stu_sub_test",
            "student_name": "Студент Тест",
            "variant": 1,
            "overall_score": 6.0,
            "max_score": 8.0,
            "final_grade": 4,
            "questions_results": [{"task_id": "tsk_1", "is_correct": True, "score": 1.0}],
            "teacher_id": "teacher_sub_1",
        }

        # First insert
        sub1, is_created = await repo.save_or_update(submission_data=payload)
        assert is_created is True
        assert sub1.client_submission_uuid == client_uuid
        assert sub1.final_grade == 4

        # Second update with same client_uuid (idempotency check)
        payload["overall_score"] = 8.0
        payload["final_grade"] = 5
        sub2, is_created2 = await repo.save_or_update(submission_data=payload)
        assert is_created2 is False
        assert sub2.submission_id == sub1.submission_id
        assert sub2.final_grade == 5

        # Query submissions
        results = await repo.query_submissions(class_id="cls_sub_test")
        assert len(results) == 1
        assert results[0].client_submission_uuid == client_uuid

        # Batch create
        batch_ins, batch_upd = await repo.batch_create([
            {
                "client_submission_uuid": client_uuid,
                "assignment_id": "TAT-07-01",
                "class_id": "cls_sub_test",
                "student_id": "stu_sub_test",
                "overall_score": 8.0,
            },
            {
                "client_submission_uuid": "new-client-uuid-123",
                "assignment_id": "TAT-07-01",
                "class_id": "cls_sub_test",
                "student_id": "stu_sub_test_2",
                "student_name": "Студент 2",
            },
        ])
        assert batch_ins == 1
        assert batch_upd == 1

        # Clear student submissions
        cleared_stu = await repo.clear_student_submissions("stu_sub_test")
        assert cleared_stu == 1

        # Clear teacher submissions
        cleared_tch = await repo.clear_teacher_submissions("teacher_sub_1")
        assert cleared_tch >= 0


# --------------------------------------------------------------------------
# 10. AssignmentRepository Tests
# --------------------------------------------------------------------------
class TestAssignmentRepository:
    @pytest.mark.asyncio
    async def test_assignment_linking(self, db_session: AsyncSession):
        c_repo = ClassRepository(db_session)
        await c_repo.get_or_create_class("cls_assign_test", name="10-А")

        a_repo = AssignmentRepository(db_session)

        # Seed assembled test
        test_model = AssembledTestModel(
            test_id="TAT-M1-TEST",
            title="Татар теленнән тест",
            grade_level=10,
            total_variants=2,
            bundle_json="{}",
        )
        db_session.add(test_model)
        await db_session.commit()

        # Assign test to class
        ca = await a_repo.create_assignment(
            class_id="cls_assign_test",
            assignment_id="TAT-M1-TEST",
            due_date="2026-12-31T23:59:59Z",
            status="active",
        )
        assert ca.class_id == "cls_assign_test"
        assert ca.assignment_id == "TAT-M1-TEST"

        # Lookup
        ca_get = await a_repo.get_class_assignment("cls_assign_test", "TAT-M1-TEST")
        assert ca_get is not None
        assert ca_get.status == "active"

        # List by class
        assignments = await a_repo.list_by_class("cls_assign_test")
        assert len(assignments) == 1

        # List with stats
        stats = await a_repo.list_assignments_with_stats(class_id="cls_assign_test")
        assert len(stats) >= 1
        found_stat = next((s for s in stats if s["assignment_id"] == "TAT-M1-TEST"), None)
        assert found_stat is not None
        assert len(found_stat["assigned_classes"]) == 1

        # Delete assignment
        del_res = await a_repo.delete_assignment("cls_assign_test", "TAT-M1-TEST")
        assert del_res is True
        assert await a_repo.get_class_assignment("cls_assign_test", "TAT-M1-TEST") is None


# --------------------------------------------------------------------------
# 11. AuditRepository Tests
# --------------------------------------------------------------------------
class TestAuditRepository:
    @pytest.mark.asyncio
    async def test_audit_logging_and_filtering(self, db_session: AsyncSession):
        repo = AuditRepository(db_session)

        # Log event
        log = await repo.log_event(
            action="USER_FREEZE",
            user_id="usr_admin_1",
            user_email="admin@test.com",
            role="admin",
            target_entity="user",
            target_id="usr_target_1",
            ip_address="127.0.0.1",
            details={"reason": "Terms violation"},
        )
        assert log.id.startswith("aud_")
        assert log.action == "USER_FREEZE"
        assert "Terms violation" in log.details_json

        # Query logs
        logs, total = await repo.query_logs(action="USER_FREEZE", user_id="usr_admin_1")
        assert total >= 1
        assert any(l.id == log.id for l in logs)

        # Non-matching action query
        logs_empty, total_empty = await repo.query_logs(action="NON_EXISTENT_ACTION")
        assert total_empty == 0
        assert len(logs_empty) == 0


# --------------------------------------------------------------------------
# 12. OAuthRepository Tests
# --------------------------------------------------------------------------
class TestOAuthRepository:
    @pytest.mark.asyncio
    async def test_oauth_linking(self, db_session: AsyncSession):
        u_repo = UserRepository(db_session)
        user = await u_repo.create_user(
            email="oauth_test@example.com",
            full_name="OAuth Иясе",
        )

        oa_repo = OAuthRepository(db_session)

        # Link account
        acc = await oa_repo.link_oauth_account(
            user_id=user.id,
            provider="yandex",
            provider_user_id="yandex_uid_12345",
            provider_email="oauth_test@yandex.ru",
        )
        assert acc.id.startswith("oa_")
        assert acc.provider == "yandex"

        # Lookup by provider and id
        by_prov = await oa_repo.get_by_provider_and_id("yandex", "yandex_uid_12345")
        assert by_prov is not None
        assert by_prov.user_id == user.id

        # List by user
        user_accs = await oa_repo.get_by_user(user.id)
        assert len(user_accs) == 1

        # Unlink account
        unlinked = await oa_repo.unlink_account(user.id, "yandex")
        assert unlinked is True
        assert (await oa_repo.get_by_provider_and_id("yandex", "yandex_uid_12345")) is None


# --------------------------------------------------------------------------
# 13. TeacherRepository Tests
# --------------------------------------------------------------------------
class TestTeacherRepository:
    @pytest.mark.asyncio
    async def test_teacher_profile_and_handshake(self, db_session: AsyncSession):
        repo = TeacherRepository(db_session)
        teacher_uuid = "device_install_uuid_abc_123"

        # Handshake upsert
        teacher = await repo.upsert_handshake(
            teacher_uuid=teacher_uuid,
            device_os="android",
            app_version="1.0.4",
            teacher_name="Сафиуллина Алсу",
            school_name="Лицей №1",
            grading_scale={"grade_5_min_pct": 85, "grade_4_min_pct": 70, "grade_3_min_pct": 50},
            confidence_threshold=0.65,
        )
        assert teacher.teacher_uuid == teacher_uuid
        assert teacher.grade_5_min_pct == 85

        # Update profile settings
        updated = await repo.update_profile_settings(
            teacher_uuid,
            teacher_name="Сафиуллина Алсу Равилевна",
            grading_scale={"grade_5_min_pct": 90},
        )
        assert updated is not None
        assert updated.teacher_name == "Сафиуллина Алсу Равилевна"
        assert updated.grade_5_min_pct == 90

        # Lookup by UUID
        assert await repo.get_by_uuid(teacher_uuid) is not None

        # Create profile linked to user
        u_repo = UserRepository(db_session)
        user = await u_repo.create_user(
            email="teacher_profile@test.com",
            full_name="Гыйззәтуллин Фәрит",
            school_name="Мәктәп №15",
        )
        user_prof = await repo.create_profile_for_user(user)
        assert user_prof.user_id == user.id
        assert user_prof.teacher_name == "Гыйззәтуллин Фәрит"
        assert await repo.get_by_user_id(user.id) is not None


# --------------------------------------------------------------------------
# 14. Transaction Safety and Concurrency Tests
# --------------------------------------------------------------------------
class TestTransactionSafety:
    @pytest.mark.asyncio
    async def test_multi_repo_atomic_transaction(self, db_session: AsyncSession):
        u_repo = UserRepository(db_session)
        a_repo = AuditRepository(db_session)

        # Coordinated transaction with auto_commit=False
        user = await u_repo.create_user(
            email="atomic@test.com",
            full_name="Atomic Transaction User",
            auto_commit=False,
        )
        log = await a_repo.log_event(
            action="CREATE_USER_ATOMIC",
            user_id=user.id,
            details={"email": user.email},
            auto_commit=False,
        )

        # Rollback everything
        await db_session.rollback()

        # Both records must NOT exist
        assert await u_repo.get_by_email("atomic@test.com") is None
        logs, _ = await a_repo.query_logs(action="CREATE_USER_ATOMIC")
        assert len(logs) == 0
