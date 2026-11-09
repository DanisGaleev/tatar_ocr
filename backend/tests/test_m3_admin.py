import json
import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
from sqlalchemy import select

from app.main import app, seed_initial_data
from app.core.config import settings
from app.core.database import Base, get_db
from app.core.security import get_password_hash, create_access_token
from app.core.telemetry import telemetry_manager
from app.models.user import UserModel
from app.models.school import ClassModel, StudentModel, ClassAssignmentModel
from app.models.submission import SubmissionModel
from app.models.teacher import TeacherModel
from app.models.session import UserSessionModel, generate_session_id
from app.models.audit import AuditLogModel
from app.repositories.user_repo import UserRepository
from app.repositories.session_repo import SessionRepository
from app.repositories.audit_repo import AuditRepository
from app.repositories.class_repo import ClassRepository
from app.repositories.student_repo import StudentRepository
from app.repositories.submission_repo import SubmissionRepository
from app.services.admin_service import AdminService
from app.schemas.admin import (
    AddStudentRequest,
    AdminFreezeUserRequest,
    AdminResetPasswordRequest,
    AdminUpdateProfileRequest,
)

M3_TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

m3_engine = create_async_engine(
    M3_TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
M3SessionLocal = async_sessionmaker(
    bind=m3_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def override_m3_get_db():
    async with M3SessionLocal() as session:
        yield session


@pytest_asyncio.fixture(autouse=True)
async def setup_m3_database():
    app.dependency_overrides[get_db] = override_m3_get_db
    app.state.session_factory = M3SessionLocal
    async with m3_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with m3_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    app.dependency_overrides.pop(get_db, None)
    if hasattr(app.state, "session_factory"):
        delattr(app.state, "session_factory")


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest_asyncio.fixture
async def db_session():
    async with M3SessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def admin_user(db_session: AsyncSession):
    repo = UserRepository(db_session)
    user = await repo.create_user(
        user_id="usr_admin_test_01",
        email="superadmin@test.ru",
        hashed_password=get_password_hash("AdminPass2026!"),
        full_name="Главный Администратор",
        school_name="Минобрнауки РТ",
        role="admin",
        status="active",
        email_verified=True,
    )
    return user


@pytest_asyncio.fixture
async def teacher_user(db_session: AsyncSession):
    repo = UserRepository(db_session)
    user = await repo.create_user(
        user_id="usr_teacher_test_01",
        email="teacher_m3@test.ru",
        hashed_password=get_password_hash("TeacherPass2026!"),
        full_name="Гөлнара Каримова",
        school_name="Гимназия №2",
        role="teacher",
        status="active",
        email_verified=True,
    )
    # Also create teacher preferences record
    t_model = TeacherModel(
        user_id=user.id,
        teacher_uuid=user.id,
        teacher_name=user.full_name,
        school_name=user.school_name,
        grade_5_min_pct=85,
        grade_4_min_pct=70,
        grade_3_min_pct=50,
        confidence_flag_threshold=0.65,
    )
    db_session.add(t_model)
    await db_session.commit()
    return user


@pytest_asyncio.fixture
async def teacher_class(db_session: AsyncSession, teacher_user: UserModel):
    cls = ClassModel(
        class_id="cls_m3_test",
        teacher_uuid=teacher_user.id,
        name="7-Б",
        subject="tatar_language",
        academic_year="2026-2027",
    )
    db_session.add(cls)
    await db_session.commit()
    return cls


@pytest_asyncio.fixture
async def teacher_student(db_session: AsyncSession, teacher_class: ClassModel):
    stu_repo = StudentRepository(db_session)
    student = await stu_repo.create_student(
        class_id=teacher_class.class_id,
        last_name="Хәйруллин",
        first_name="Илнур",
        middle_name="Фәрит улы",
        student_id="stu_m3_01",
    )
    return student


def admin_headers(admin_user: UserModel):
    token = create_access_token(data={"sub": admin_user.id, "role": admin_user.role})
    return {"Authorization": f"Bearer {token}"}


def teacher_headers(teacher_user: UserModel):
    token = create_access_token(data={"sub": teacher_user.id, "role": teacher_user.role})
    return {"Authorization": f"Bearer {token}"}


# =============================================================================
# 1. Non-Admin Security Rejection & Unauthenticated Tests
# =============================================================================
class TestAdminAccessControl:
    @pytest.mark.asyncio
    async def test_unauthenticated_request_rejected_401(self, client: AsyncClient):
        res = await client.get("/api/v1/admin/users")
        assert res.status_code == 401
        data = res.json()
        assert data.get("error_code") == "AUTH_ERROR"

    @pytest.mark.asyncio
    async def test_teacher_role_rejected_403(self, client: AsyncClient, teacher_user: UserModel):
        res = await client.get("/api/v1/admin/users", headers=teacher_headers(teacher_user))
        assert res.status_code == 403
        data = res.json()
        assert data.get("error_code") == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_frozen_user_rejected_403(
        self, client: AsyncClient, admin_user: UserModel, db_session: AsyncSession
    ):
        # Freeze admin
        user_repo = UserRepository(db_session)
        await user_repo.set_status(admin_user, "frozen")

        res = await client.get("/api/v1/admin/users", headers=admin_headers(admin_user))
        assert res.status_code == 403
        data = res.json()
        assert "frozen" in data.get("detail", "").lower()


# =============================================================================
# 2. Users Management: List & User Detail Cards
# =============================================================================
class TestAdminUserManagement:
    @pytest.mark.asyncio
    async def test_list_users_paginated_and_filters(
        self, client: AsyncClient, admin_user: UserModel, teacher_user: UserModel
    ):
        res = await client.get(
            "/api/v1/admin/users?page=1&page_size=10",
            headers=admin_headers(admin_user),
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total"] >= 2
        assert len(data["users"]) >= 2

        # Filter by role
        res_role = await client.get(
            "/api/v1/admin/users?role=teacher",
            headers=admin_headers(admin_user),
        )
        assert res_role.status_code == 200
        teachers = res_role.json()["users"]
        assert all(u["role"] == "teacher" for u in teachers)

        # Filter by search
        res_search = await client.get(
            "/api/v1/admin/users?search=Гөлнара",
            headers=admin_headers(admin_user),
        )
        assert res_search.status_code == 200
        matched = res_search.json()["users"]
        assert len(matched) == 1
        assert matched[0]["id"] == teacher_user.id

    @pytest.mark.asyncio
    async def test_get_user_detail_card_with_stats(
        self,
        client: AsyncClient,
        admin_user: UserModel,
        teacher_user: UserModel,
        teacher_class: ClassModel,
        teacher_student: StudentModel,
        db_session: AsyncSession,
    ):
        # Add a submission for this teacher
        sub = SubmissionModel(
            submission_id="sub_test_detail",
            teacher_id=teacher_user.id,
            class_id=teacher_class.class_id,
            student_id=teacher_student.student_id,
            student_name="Хәйруллин Илнур",
            assignment_id="TAT-2026-Q1",
            final_grade=5,
            questions_results_json="[]",
            checked_at=datetime.now(timezone.utc),
        )
        db_session.add(sub)

        # Add an active session
        ses = UserSessionModel(
            id=generate_session_id(),
            user_id=teacher_user.id,
            refresh_token_hash="hash_detail_session",
            device_info="Chrome on Linux",
            ip_address="192.168.1.10",
            user_agent="Mozilla/5.0",
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )
        db_session.add(ses)
        await db_session.commit()

        res = await client.get(
            f"/api/v1/admin/users/{teacher_user.id}",
            headers=admin_headers(admin_user),
        )
        assert res.status_code == 200
        data = res.json()
        assert data["user"]["id"] == teacher_user.id
        assert data["stats"]["classes_count"] == 1
        assert data["stats"]["students_count"] == 1
        assert data["stats"]["submissions_count"] == 1
        assert data["stats"]["average_grade"] == 5.0
        assert len(data["recent_sessions"]) == 1
        assert data["preferences"] is not None
        assert data["preferences"]["grading_scale"]["grade_5_min_pct"] == 85

    @pytest.mark.asyncio
    async def test_get_nonexistent_user_returns_404(
        self, client: AsyncClient, admin_user: UserModel
    ):
        res = await client.get(
            "/api/v1/admin/users/usr_nonexistent_999",
            headers=admin_headers(admin_user),
        )
        assert res.status_code == 404
        assert res.json().get("error_code") == "NOT_FOUND"


# =============================================================================
# 3. Security Actions: Freeze, Unfreeze, Self-Harm Protection
# =============================================================================
class TestAdminFreezeUnfreeze:
    @pytest.mark.asyncio
    async def test_freeze_and_unfreeze_lifecycle(
        self,
        client: AsyncClient,
        admin_user: UserModel,
        teacher_user: UserModel,
        db_session: AsyncSession,
    ):
        # 1. Create active session for teacher
        ses_repo = SessionRepository(db_session)
        ses = await ses_repo.create_session(
            user_id=teacher_user.id,
            refresh_token_hash="token_hash_freeze_test",
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
        assert ses.revoked_at is None

        # 2. Freeze teacher
        freeze_res = await client.post(
            f"/api/v1/admin/users/{teacher_user.id}/freeze",
            headers=admin_headers(admin_user),
            json={"reason": "Security audit suspicion"},
        )
        assert freeze_res.status_code == 200
        freeze_data = freeze_res.json()
        assert freeze_data["status"] == "frozen"
        assert freeze_data["sessions_revoked"] == 1

        # Verify teacher cannot access me or protected endpoints
        t_res = await client.get("/api/v1/auth/me", headers=teacher_headers(teacher_user))
        assert t_res.status_code == 403

        # 3. Freezing already frozen user returns 400
        dup_freeze = await client.post(
            f"/api/v1/admin/users/{teacher_user.id}/freeze",
            headers=admin_headers(admin_user),
        )
        assert dup_freeze.status_code == 400
        assert dup_freeze.json().get("error_code") == "BUSINESS_VALIDATION_ERROR"

        # 4. Unfreeze teacher
        unfreeze_res = await client.post(
            f"/api/v1/admin/users/{teacher_user.id}/unfreeze",
            headers=admin_headers(admin_user),
        )
        assert unfreeze_res.status_code == 200
        assert unfreeze_res.json()["status"] == "active"

        # 5. Unfreezing active user returns 400
        dup_unfreeze = await client.post(
            f"/api/v1/admin/users/{teacher_user.id}/unfreeze",
            headers=admin_headers(admin_user),
        )
        assert dup_unfreeze.status_code == 400

    @pytest.mark.asyncio
    async def test_self_freeze_prohibited_400(
        self, client: AsyncClient, admin_user: UserModel
    ):
        res = await client.post(
            f"/api/v1/admin/users/{admin_user.id}/freeze",
            headers=admin_headers(admin_user),
        )
        assert res.status_code == 400
        assert "own account" in res.json().get("detail", "").lower()


# =============================================================================
# 4. Identity & Profile: Reset Password, Patch Profile, Delete User
# =============================================================================
class TestAdminIdentityAndProfile:
    @pytest.mark.asyncio
    async def test_reset_password_manual_and_auto(
        self, client: AsyncClient, admin_user: UserModel, teacher_user: UserModel
    ):
        # Reset with manual password
        res_manual = await client.post(
            f"/api/v1/admin/users/{teacher_user.id}/reset-password",
            headers=admin_headers(admin_user),
            json={"new_password": "NewManualPassword123!"},
        )
        assert res_manual.status_code == 200
        assert res_manual.json()["status"] == "password_reset"
        assert res_manual.json()["temporary_password"] == "NewManualPassword123!"

        # Reset with password shorter than 8 chars returns 400
        res_short = await client.post(
            f"/api/v1/admin/users/{teacher_user.id}/reset-password",
            headers=admin_headers(admin_user),
            json={"new_password": "short"},
        )
        assert res_short.status_code == 400

        # Reset with omitted password -> auto-generates 12-char temp password
        res_auto = await client.post(
            f"/api/v1/admin/users/{teacher_user.id}/reset-password",
            headers=admin_headers(admin_user),
            json={},
        )
        assert res_auto.status_code == 200
        temp_pwd = res_auto.json()["temporary_password"]
        assert len(temp_pwd) == 12

    @pytest.mark.asyncio
    async def test_patch_user_profile(
        self, client: AsyncClient, admin_user: UserModel, teacher_user: UserModel
    ):
        res = await client.patch(
            f"/api/v1/admin/users/{teacher_user.id}/profile",
            headers=admin_headers(admin_user),
            json={
                "full_name": "Каримова Гөлнара Илдар кызы (Обновлено)",
                "school_name": "Лицей №131",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["full_name"] == "Каримова Гөлнара Илдар кызы (Обновлено)"
        assert data["school_name"] == "Лицей №131"

    @pytest.mark.asyncio
    async def test_soft_delete_user_and_self_delete_guard(
        self, client: AsyncClient, admin_user: UserModel, teacher_user: UserModel
    ):
        # Self-delete prohibited
        self_del = await client.delete(
            f"/api/v1/admin/users/{admin_user.id}",
            headers=admin_headers(admin_user),
        )
        assert self_del.status_code == 400

        # Delete teacher
        del_res = await client.delete(
            f"/api/v1/admin/users/{teacher_user.id}",
            headers=admin_headers(admin_user),
        )
        assert del_res.status_code == 200
        assert del_res.json()["status"] == "deleted"

        # Verify teacher status is deleted and PII is anonymized
        detail_res = await client.get(
            f"/api/v1/admin/users/{teacher_user.id}",
            headers=admin_headers(admin_user),
        )
        assert detail_res.status_code == 200
        user_info = detail_res.json()["user"]
        assert user_info["status"] == "deleted"
        assert "Удаленный" in user_info["full_name"]


# =============================================================================
# 5. Teacher Data Purge: Clear Teacher Data
# =============================================================================
class TestAdminDataPurge:
    @pytest.mark.asyncio
    async def test_clear_teacher_data_purges_classes_and_preserves_user(
        self,
        client: AsyncClient,
        admin_user: UserModel,
        teacher_user: UserModel,
        teacher_class: ClassModel,
        teacher_student: StudentModel,
        db_session: AsyncSession,
    ):
        # Add submission
        sub = SubmissionModel(
            submission_id="sub_purge_test",
            teacher_id=teacher_user.id,
            class_id=teacher_class.class_id,
            student_id=teacher_student.student_id,
            student_name="Хәйруллин Илнур",
            assignment_id="TAT-2026-Q1",
            final_grade=4,
            questions_results_json="[]",
            checked_at=datetime.now(timezone.utc),
        )
        db_session.add(sub)
        await db_session.commit()

        # Perform clear-data
        res = await client.post(
            f"/api/v1/admin/users/{teacher_user.id}/clear-data",
            headers=admin_headers(admin_user),
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "data_cleared"
        assert data["deleted_classes_count"] == 1
        assert data["deleted_submissions_count"] == 1

        # Verify teacher user still exists with status active
        user_repo = UserRepository(db_session)
        user_check = await user_repo.get_by_id(teacher_user.id)
        assert user_check is not None
        assert user_check.status == "active"

        # Verify classes are gone
        cls_repo = ClassRepository(db_session)
        classes = await cls_repo.list_by_teacher(teacher_user.id)
        assert len(classes) == 0


# =============================================================================
# 6. Student Roster Management under Teacher Classes
# =============================================================================
class TestAdminStudentManagement:
    @pytest.mark.asyncio
    async def test_add_delete_clear_student(
        self,
        client: AsyncClient,
        admin_user: UserModel,
        teacher_user: UserModel,
        teacher_class: ClassModel,
        db_session: AsyncSession,
    ):
        # 1. Add student
        add_res = await client.post(
            f"/api/v1/admin/users/{teacher_user.id}/classes/{teacher_class.class_id}/students",
            headers=admin_headers(admin_user),
            json={
                "last_name": "Нигмәтуллин",
                "first_name": "Айдар",
                "middle_name": "Рөстәм улы",
                "student_id": "stu_admin_created_01",
            },
        )
        assert add_res.status_code == 201
        data = add_res.json()
        assert data["student_id"] == "stu_admin_created_01"
        assert data["last_name"] == "Нигмәтуллин"
        assert data["first_name"] == "Айдар"

        # 2. Add submission for this student
        sub = SubmissionModel(
            submission_id="sub_stu_clear_test",
            teacher_id=teacher_user.id,
            class_id=teacher_class.class_id,
            student_id="stu_admin_created_01",
            student_name="Нигмәтуллин Айдар",
            assignment_id="TAT-2026-Q1",
            final_grade=5,
            questions_results_json="[]",
            checked_at=datetime.now(timezone.utc),
        )
        db_session.add(sub)
        await db_session.commit()

        # 3. Clear student data
        clear_res = await client.post(
            f"/api/v1/admin/users/{teacher_user.id}/classes/{teacher_class.class_id}/students/stu_admin_created_01/clear-data",
            headers=admin_headers(admin_user),
        )
        assert clear_res.status_code == 200
        assert clear_res.json()["deleted_submissions_count"] == 1

        # 4. Delete student from class
        del_res = await client.delete(
            f"/api/v1/admin/users/{teacher_user.id}/classes/{teacher_class.class_id}/students/stu_admin_created_01",
            headers=admin_headers(admin_user),
        )
        assert del_res.status_code == 200
        assert del_res.json()["status"] == "deleted"
        assert del_res.json()["is_active"] is False

    @pytest.mark.asyncio
    async def test_add_student_wrong_teacher_returns_404(
        self, client: AsyncClient, admin_user: UserModel, teacher_class: ClassModel
    ):
        res = await client.post(
            f"/api/v1/admin/users/usr_different_teacher/classes/{teacher_class.class_id}/students",
            headers=admin_headers(admin_user),
            json={"last_name": "Алиев", "first_name": "Руслан"},
        )
        assert res.status_code == 404


# =============================================================================
# 7. Audit Logs and Session Management
# =============================================================================
class TestAdminAuditAndSessions:
    @pytest.mark.asyncio
    async def test_audit_logs_query_and_filtering(
        self, client: AsyncClient, admin_user: UserModel, db_session: AsyncSession
    ):
        audit_repo = AuditRepository(db_session)
        await audit_repo.log_event(
            action="FREEZE_USER",
            user_id=admin_user.id,
            user_email=admin_user.email,
            role="admin",
            target_entity="user",
            target_id="usr_target_123",
            details={"reason": "test audit"},
        )
        await audit_repo.log_event(
            action="LOGIN",
            user_id=admin_user.id,
            user_email=admin_user.email,
            role="admin",
            target_entity="auth",
            details={"ip": "127.0.0.1"},
        )

        res = await client.get("/api/v1/admin/audit-logs", headers=admin_headers(admin_user))
        assert res.status_code == 200
        data = res.json()
        assert data["total"] >= 2
        assert len(data["logs"]) >= 2

        # Filter by action
        res_filter = await client.get(
            "/api/v1/admin/audit-logs?action=FREEZE_USER",
            headers=admin_headers(admin_user),
        )
        assert res_filter.status_code == 200
        filtered_logs = res_filter.json()["logs"]
        assert all(l["action"] == "FREEZE_USER" for l in filtered_logs)

    @pytest.mark.asyncio
    async def test_active_sessions_list_and_kill(
        self, client: AsyncClient, admin_user: UserModel, teacher_user: UserModel, db_session: AsyncSession
    ):
        ses_repo = SessionRepository(db_session)
        ses = await ses_repo.create_session(
            user_id=teacher_user.id,
            refresh_token_hash="hash_kill_test",
            device_info="Firefox on macOS",
            ip_address="10.0.0.1",
            user_agent="Mozilla/5.0",
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        )

        # 1. List active sessions
        res = await client.get("/api/v1/admin/active-sessions", headers=admin_headers(admin_user))
        assert res.status_code == 200
        data = res.json()
        assert data["total"] >= 1
        found = any(s["id"] == ses.id for s in data["sessions"])
        assert found is True

        # 2. Kill session
        kill_res = await client.delete(
            f"/api/v1/admin/active-sessions/{ses.id}",
            headers=admin_headers(admin_user),
        )
        assert kill_res.status_code == 200
        assert kill_res.json()["status"] == "revoked"

        # 3. Kill already revoked session returns 404
        kill_again = await client.delete(
            f"/api/v1/admin/active-sessions/{ses.id}",
            headers=admin_headers(admin_user),
        )
        assert kill_again.status_code == 404


# =============================================================================
# 8. Real-Time Telemetry: WebSocket Live Feed
# =============================================================================
class TestAdminWebSocketLiveFeed:
    def test_ws_unauthenticated_rejected_with_1008(self):
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/api/v1/admin/ws/live-feed"):
                pass
        assert exc.value.code == 1008

    def test_ws_teacher_role_rejected_with_1008(self, teacher_user: UserModel):
        teacher_token = create_access_token(data={"sub": teacher_user.id, "role": "teacher"})
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(f"/api/v1/admin/ws/live-feed?token={teacher_token}"):
                pass
        assert exc.value.code == 1008

    def test_ws_admin_connected_ping_pong_and_telemetry(self, admin_user: UserModel):
        token = create_access_token(data={"sub": admin_user.id, "role": "admin"})
        client = TestClient(app)
        with client.websocket_connect(f"/api/v1/admin/ws/live-feed?token={token}") as ws:
            # 1. Ping / Pong
            ws.send_text(json.dumps({"type": "ping"}))
            resp = ws.receive_json()
            assert resp.get("type") == "pong"
            assert "timestamp" in resp


# =============================================================================
# 9. Lifespan Default Admin Seeder Verification
# =============================================================================
class TestLifespanAdminSeeder:
    @pytest.mark.asyncio
    async def test_seed_initial_data_creates_admin_if_missing(self, db_session: AsyncSession):
        await seed_initial_data(db_session)
        user_repo = UserRepository(db_session)
        admin = await user_repo.get_by_email(settings.ADMIN_EMAIL)
        assert admin is not None
        assert admin.role == "admin"
        assert admin.status == "active"
        assert admin.email_verified is True
