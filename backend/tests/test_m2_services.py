import json
import pytest
import pytest_asyncio
import pyotp
from datetime import datetime, timezone
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
from sqlalchemy import select

from app.main import app
from app.core.config import settings
from app.core.database import Base, get_db
from app.core.dependencies import (
    get_current_active_user,
    require_admin,
    require_superadmin,
    get_owned_class,
)
from app.core.exceptions import (
    AuthError,
    PermissionDeniedError,
    NotFoundError,
    BusinessValidationError,
    RateLimitError,
)
from app.core.security import get_password_hash, create_access_token
from app.core.email import email_service
from app.models.user import UserModel
from app.models.school import ClassModel, StudentModel
from app.models.test import AssembledTestModel
from app.models.submission import SubmissionModel
from app.models.teacher import TeacherModel
from app.models.session import UserSessionModel

from app.services.auth_service import AuthService
from app.services.classes_service import ClassesService
from app.services.submission_service import SubmissionService
from app.services.teacher_service import TeacherService

from app.routers.classes import (
    AddStudentRequest,
    UpdateStudentRequest,
)
from app.schemas.classes import (
    AssignTestToClassRequest,
)
from app.schemas.submission import (
    SingleSubmissionSaveRequest,
    SubmissionItem,
    BatchSyncSubmissionsRequest,
)
from app.schemas.user import (
    TeacherProfileUpdateRequest,
    TeacherChangePasswordRequest,
    GradingScale,
)

M2_TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

m2_engine = create_async_engine(
    M2_TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
M2SessionLocal = async_sessionmaker(
    bind=m2_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def override_m2_get_db():
    async with M2SessionLocal() as session:
        yield session


@pytest_asyncio.fixture(autouse=True)
async def setup_m2_database():
    app.dependency_overrides[get_db] = override_m2_get_db
    async with m2_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with m2_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest_asyncio.fixture
async def db_session():
    async with M2SessionLocal() as session:
        yield session


# =============================================================================
# 1. AuthService Unit & Integration Tests
# =============================================================================
class TestAuthService:
    @pytest.mark.asyncio
    async def test_register_creates_unverified_and_sends_otp(self, db_session: AsyncSession):
        email_service.clear_sent_codes()
        service = AuthService(db_session)
        res = await service.register(
            email="teacher_m2_new@tatar.ru",
            password="StrongPassword2026!",
            full_name="Гөлнара Каримова",
            school_name="Гимназия №2",
        )
        assert res.status == "unverified"
        assert res.email == "teacher_m2_new@tatar.ru"

        code = email_service.get_last_code_for("teacher_m2_new@tatar.ru", "register_verify")
        assert code is not None
        assert len(code) == 6

    @pytest.mark.asyncio
    async def test_register_weak_password_rejected(self, db_session: AsyncSession):
        service = AuthService(db_session)
        with pytest.raises(BusinessValidationError) as exc:
            await service.register(
                email="weak_pwd@tatar.ru",
                password="short",
                full_name="Тест",
            )
        assert "at least 8 characters" in str(exc.value.message)

    @pytest.mark.asyncio
    async def test_verify_email_activates_user_and_issues_tokens(self, db_session: AsyncSession):
        email_service.clear_sent_codes()
        service = AuthService(db_session)
        email = "verify_m2@tatar.ru"
        await service.register(
            email=email,
            password="StrongPassword2026!",
            full_name="Камилә Сәлихова",
            school_name="Мәктәп №15",
        )
        code = email_service.get_last_code_for(email, "register_verify")
        assert code is not None

        # Verify wrong code decrements attempts
        with pytest.raises(BusinessValidationError) as exc_bad:
            await service.verify_email(email=email, code="000000")
        assert "Attempts remaining" in str(exc_bad.value.message)

        # Verify correct code succeeds
        tokens = await service.verify_email(email=email, code=code)
        assert tokens.access_token is not None
        assert tokens.refresh_token is not None
        assert tokens.user.status == "active"
        assert tokens.user.email_verified is True

    @pytest.mark.asyncio
    async def test_resend_code_cooldown(self, db_session: AsyncSession):
        email_service.clear_sent_codes()
        service = AuthService(db_session)
        email = "cooldown@tatar.ru"
        await service.register(
            email=email,
            password="StrongPassword2026!",
            full_name="Ренат",
        )
        # Attempting immediate resend triggers RateLimitError (429)
        with pytest.raises(RateLimitError) as exc_rate:
            await service.resend_code(email=email)
        assert exc_rate.value.status_code == 429
        assert "Please wait" in str(exc_rate.value.message)

    @pytest.mark.asyncio
    async def test_login_brute_force_lockout_after_5_attempts(self, db_session: AsyncSession):
        email_service.clear_sent_codes()
        service = AuthService(db_session)
        email = "lockout@tatar.ru"
        await service.register(
            email=email,
            password="CorrectPassword123!",
            full_name="Алмаз",
        )
        code = email_service.get_last_code_for(email, "register_verify")
        await service.verify_email(email=email, code=code)

        # 4 wrong password attempts raise AuthError (401)
        for _ in range(4):
            with pytest.raises(AuthError):
                await service.login(email=email, password="WrongPassword")

        # 5th failed attempt triggers brute force lockout (403 PermissionDeniedError)
        with pytest.raises(PermissionDeniedError) as exc_lock:
            await service.login(email=email, password="WrongPassword")
        assert "locked" in str(exc_lock.value.message).lower()

    @pytest.mark.asyncio
    async def test_2fa_totp_lifecycle(self, db_session: AsyncSession):
        email_service.clear_sent_codes()
        service = AuthService(db_session)
        email = "totp_user@tatar.ru"
        await service.register(
            email=email,
            password="Password123!",
            full_name="Зөһрә",
        )
        code = email_service.get_last_code_for(email, "register_verify")
        login_res = await service.verify_email(email=email, code=code)
        user_repo = service._user_repo
        user = await user_repo.get_by_email(email)

        # Setup 2FA
        setup_res = await service.setup_2fa(user)
        assert setup_res.secret is not None
        assert setup_res.qr_code_base64.startswith("data:image/png;base64,")

        # Confirm 2FA with current TOTP code
        current_totp = pyotp.TOTP(setup_res.secret).now()
        confirm_res = await service.confirm_2fa_setup(user, current_totp)
        assert confirm_res.two_factor_enabled is True
        assert len(confirm_res.backup_codes) == 5

        # Login now challenges for 2FA
        login_challenge = await service.login(email=email, password="Password123!")
        assert hasattr(login_challenge, "requires_2fa") and login_challenge.requires_2fa is True
        temp_token = login_challenge.temp_token

        # Verify 2FA with temp token
        current_totp_2 = pyotp.TOTP(setup_res.secret).now()
        full_tokens = await service.verify_2fa(temp_token=temp_token, code=current_totp_2)
        assert full_tokens.access_token is not None
        assert full_tokens.user.email == email

        # Disable 2FA with password
        await service.disable_2fa(user, "Password123!")
        user_refreshed = await user_repo.get_by_email(email)
        assert user_refreshed.two_factor_enabled is False

    @pytest.mark.asyncio
    async def test_token_rotation_and_revocation(self, db_session: AsyncSession):
        email_service.clear_sent_codes()
        service = AuthService(db_session)
        email = "rotation@tatar.ru"
        await service.register(email=email, password="Password123!", full_name="Ирек")
        code = email_service.get_last_code_for(email, "register_verify")
        tokens = await service.verify_email(email=email, code=code)

        # Rotate tokens
        new_tokens = await service.refresh_tokens(refresh_token=tokens.refresh_token)
        assert new_tokens.access_token != tokens.access_token
        assert new_tokens.refresh_token != tokens.refresh_token

        # Old refresh token is now revoked and must be rejected
        with pytest.raises(AuthError):
            await service.refresh_tokens(refresh_token=tokens.refresh_token)

    @pytest.mark.asyncio
    async def test_reset_password_flow(self, db_session: AsyncSession):
        email_service.clear_sent_codes()
        service = AuthService(db_session)
        email = "reset_pwd@tatar.ru"
        await service.register(email=email, password="OldPassword123!", full_name="Данис")
        vcode = email_service.get_last_code_for(email, "register_verify")
        await service.verify_email(email=email, code=vcode)

        # Request forgot password OTP
        await service.forgot_password(email)
        reset_otp = email_service.get_last_code_for(email, "password_reset")
        assert reset_otp is not None

        # Reset password
        await service.reset_password(email=email, code=reset_otp, new_password="NewBrandPassword2026!")

        # Login with old password fails
        with pytest.raises(AuthError):
            await service.login(email=email, password="OldPassword123!")

        # Login with new password succeeds
        tokens = await service.login(email=email, password="NewBrandPassword2026!")
        assert tokens.access_token is not None


# =============================================================================
# 2. ClassesService Unit & Tenancy Tests
# =============================================================================
class TestClassesService:
    @pytest.mark.asyncio
    async def test_classes_and_student_roster_lifecycle(self, db_session: AsyncSession):
        classes_svc = ClassesService(db_session)
        teacher_a = UserModel(
            id="usr_teacher_a",
            email="teacher_a@tatar.ru",
            full_name="Гөлнара Каримова",
            role="teacher",
            status="active",
        )
        db_session.add(teacher_a)
        await db_session.commit()

        # Bulk import creates class and students
        roster_text = "Галиев Амир Рустемович\nСафина Дилә Нияз кызы"
        import_res = await classes_svc.bulk_import_students(
            class_id="cls_8a_test",
            raw_text=roster_text,
            current_user=teacher_a,
        )
        assert import_res.added_count == 2

        # List classes includes the imported class
        cls_list = await classes_svc.list_classes(teacher_a)
        assert any(c.class_id == "cls_8a_test" and c.student_count == 2 for c in cls_list.classes)

        # Get class students
        students_res = await classes_svc.get_class_students("cls_8a_test", teacher_a)
        assert len(students_res.students) == 2
        student1 = students_res.students[0]
        assert student1.last_name == "Галиев"

        # Add single student
        new_stu = await classes_svc.add_student(
            class_id="cls_8a_test",
            last_name="Җәлилов",
            first_name="Муса",
            middle_name="Мостафа улы",
            current_user=teacher_a,
        )
        assert new_stu.short_name == "Җәлилов М."

        # Update student
        updated_stu = await classes_svc.update_student(
            class_id="cls_8a_test",
            student_id=new_stu.student_id,
            first_name="Муса-Шагыйрь",
            current_user=teacher_a,
        )
        assert "Муса-Шагыйрь" in updated_stu.full_name

        # Soft delete student
        del_res = await classes_svc.delete_student(
            class_id="cls_8a_test",
            student_id=new_stu.student_id,
            current_user=teacher_a,
        )
        assert del_res["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_classes_idor_rejection(self, db_session: AsyncSession):
        classes_svc = ClassesService(db_session)
        teacher_owner = UserModel(
            id="usr_teacher_owner",
            email="owner@tatar.ru",
            full_name="Учитель Владелец",
            role="teacher",
            status="active",
        )
        teacher_intruder = UserModel(
            id="usr_teacher_intruder",
            email="intruder@tatar.ru",
            full_name="Учитель Взломщик",
            role="teacher",
            status="active",
        )
        cls_obj = ClassModel(
            class_id="cls_private_1",
            name="9-Б",
            teacher_uuid="usr_teacher_owner",
            subject="tatar_language",
        )
        db_session.add_all([teacher_owner, teacher_intruder, cls_obj])
        await db_session.commit()

        # Non-owner accessing class roster raises PermissionDeniedError (403)
        with pytest.raises(PermissionDeniedError) as exc:
            await classes_svc.get_class_students("cls_private_1", teacher_intruder)
        assert "Access denied" in str(exc.value.message)


# =============================================================================
# 3. SubmissionService Unit & Multi-Tenancy Tests
# =============================================================================
class TestSubmissionService:
    @pytest.mark.asyncio
    async def test_save_single_submission_idempotency(self, db_session: AsyncSession):
        sub_svc = SubmissionService(db_session)
        teacher = UserModel(
            id="usr_teacher_sub",
            email="sub_teacher@tatar.ru",
            full_name="Учитель Сабмишн",
            role="teacher",
            status="active",
        )
        cls_obj = ClassModel(
            class_id="cls_sub_1",
            name="7-В",
            teacher_uuid="usr_teacher_sub",
            subject="tatar_language",
        )
        db_session.add_all([teacher, cls_obj])
        await db_session.commit()

        client_uuid = "client-uuid-m2-unique-01"
        item = SubmissionItem(
            client_submission_uuid=client_uuid,
            student_id="stu_101",
            student_name="Галиев Амир",
            variant=1,
            checked_at=datetime.now(timezone.utc),
            overall_score=7.0,
            max_score=8.0,
            final_grade=5,
            teacher_reviewed_flags=0,
            questions_results=[],
        )
        req = SingleSubmissionSaveRequest(
            class_id="cls_sub_1",
            assignment_id="ASSGN-01",
            submission=item,
        )

        # First save
        res1 = await sub_svc.save_single_submission(req, teacher)
        assert res1.status == "saved"
        assert res1.submission_id is not None

        # Second save with same client UUID updates rather than creates duplicate
        item.overall_score = 8.0
        res2 = await sub_svc.save_single_submission(req, teacher)
        assert res2.submission_id == res1.submission_id

        # Query confirms score is updated
        saved_sub = await sub_svc.get_submission(res1.submission_id, teacher)
        assert saved_sub.overall_score == 8.0

    @pytest.mark.asyncio
    async def test_submission_idor_rejection(self, db_session: AsyncSession):
        sub_svc = SubmissionService(db_session)
        teacher_owner = UserModel(
            id="usr_teacher_owner_sub",
            email="owner_sub@tatar.ru",
            full_name="Владелец",
            role="teacher",
            status="active",
        )
        teacher_intruder = UserModel(
            id="usr_teacher_bad",
            email="bad@tatar.ru",
            full_name="Взломщик",
            role="teacher",
            status="active",
        )
        cls_obj = ClassModel(
            class_id="cls_sub_private",
            name="7-В",
            teacher_uuid="usr_teacher_owner_sub",
            subject="tatar_language",
        )
        db_session.add_all([teacher_owner, teacher_intruder, cls_obj])
        await db_session.commit()

        # Batch sync to unowned class raises PermissionDeniedError (403)
        req = BatchSyncSubmissionsRequest(
            class_id="cls_sub_private",
            assignment_id="ASSGN-01",
            submissions=[],
        )
        with pytest.raises(PermissionDeniedError):
            await sub_svc.batch_sync_submissions(req, teacher_intruder)


# =============================================================================
# 4. TeacherService Unit Tests
# =============================================================================
class TestTeacherService:
    @pytest.mark.asyncio
    async def test_get_and_update_profile(self, db_session: AsyncSession):
        svc = TeacherService(db_session)
        teacher_user = UserModel(
            id="usr_prof_1",
            email="prof@tatar.ru",
            full_name="Рәисә",
            hashed_password=get_password_hash("Password123!"),
            role="teacher",
            status="active",
        )
        db_session.add(teacher_user)
        await db_session.commit()

        # Profile is lazily initialized
        prof = await svc.get_profile(teacher_user)
        assert prof.user.email == "prof@tatar.ru"
        assert prof.preferences.grading_scale.grade_5_min_pct == 85

        # Update profile settings
        up_req = TeacherProfileUpdateRequest(
            full_name="Рәисә Закирова",
            school_name="Лицей №2",
            grading_scale=GradingScale(grade_5_min_pct=90, grade_4_min_pct=75, grade_3_min_pct=55),
            confidence_flag_threshold=0.72,
        )
        updated = await svc.update_profile(teacher_user, up_req)
        assert updated.user.full_name == "Рәисә Закирова"
        assert updated.preferences.grading_scale.grade_5_min_pct == 90
        assert updated.preferences.confidence_flag_threshold == 0.72

    @pytest.mark.asyncio
    async def test_change_password_with_validation(self, db_session: AsyncSession):
        svc = TeacherService(db_session)
        teacher_user = UserModel(
            id="usr_pwd_chg",
            email="pwd_chg@tatar.ru",
            full_name="Фанис",
            hashed_password=get_password_hash("OldPassword123!"),
            role="teacher",
            status="active",
        )
        db_session.add(teacher_user)
        await db_session.commit()

        # Invalid old password raises BusinessValidationError
        with pytest.raises(BusinessValidationError) as exc_old:
            await svc.change_password(
                teacher_user,
                TeacherChangePasswordRequest(old_password="WrongOldPassword!", new_password="NewPassword123!"),
            )
        assert "Invalid old password" in str(exc_old.value.message)

        # Password < 8 characters raises BusinessValidationError
        with pytest.raises(BusinessValidationError) as exc_len:
            await svc.change_password(
                teacher_user,
                TeacherChangePasswordRequest.model_construct(old_password="OldPassword123!", new_password="short"),
            )
        assert "at least 8 characters" in str(exc_len.value.message)

        # Successful password change
        succ = await svc.change_password(
            teacher_user,
            TeacherChangePasswordRequest(old_password="OldPassword123!", new_password="NewPassword123!"),
        )
        assert succ["status"] == "success"


# =============================================================================
# 5. Security & FastAPI Dependency Hardening Tests (HTTP Level)
# =============================================================================
class TestSecurityAndDependencies:
    @pytest.mark.asyncio
    async def test_strict_mode_rejects_unauthenticated_requests_with_401(self, client: AsyncClient, monkeypatch):
        # Strict mode prevents unauthenticated access
        monkeypatch.setattr(settings, "AUTH_ENFORCED_STRICT", True)
        res = await client.get("/api/v1/classes")
        assert res.status_code == 401
        assert res.json()["error_code"] == "AUTH_ERROR"

    @pytest.mark.asyncio
    async def test_strict_mode_rejects_x_teacher_uuid_bypass_with_401(self, client: AsyncClient, monkeypatch):
        # Spoofed X-Teacher-UUID header without Bearer token must return 401 in strict mode
        monkeypatch.setattr(settings, "AUTH_ENFORCED_STRICT", True)
        res = await client.get("/api/v1/classes", headers={"X-Teacher-UUID": "c56a4180-65aa-42ec-a945-5fd21dec0538"})
        assert res.status_code == 401
        assert res.json()["error_code"] == "AUTH_ERROR"

    @pytest.mark.asyncio
    async def test_strict_paths_always_reject_unauthenticated(self, client: AsyncClient):
        # /auth/me strictly requires Bearer token even if AUTH_ENFORCED_STRICT is False
        res = await client.get("/api/v1/auth/me")
        assert res.status_code == 401
        assert res.json()["error_code"] == "AUTH_ERROR"

    @pytest.mark.asyncio
    async def test_frozen_user_rejected_with_403(self, client: AsyncClient, db_session: AsyncSession):
        frozen_teacher = UserModel(
            id="usr_frozen_m2",
            email="frozen_m2@tatar.ru",
            hashed_password=get_password_hash("Password123!"),
            full_name="Замороженный",
            status="frozen",
            role="teacher",
        )
        db_session.add(frozen_teacher)
        await db_session.commit()

        # Login returns 403 PermissionDeniedError
        res_login = await client.post("/api/v1/auth/login", json={"email": "frozen_m2@tatar.ru", "password": "Password123!"})
        assert res_login.status_code == 403
        assert res_login.json()["error_code"] == "PERMISSION_DENIED"

        # Direct token call returns 403 PermissionDeniedError
        token = create_access_token({"sub": "usr_frozen_m2", "email": "frozen_m2@tatar.ru", "role": "teacher"})
        res_token = await client.get("/api/v1/classes", headers={"Authorization": f"Bearer {token}"})
        assert res_token.status_code == 403
        assert res_token.json()["error_code"] == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_idor_cross_teacher_class_access_rejected_with_403(self, client: AsyncClient, db_session: AsyncSession):
        teacher_owner = UserModel(
            id="usr_owner_http",
            email="owner_http@tatar.ru",
            full_name="Владелец",
            status="active",
            role="teacher",
        )
        teacher_other = UserModel(
            id="usr_other_http",
            email="other_http@tatar.ru",
            full_name="Чужой Учитель",
            status="active",
            role="teacher",
        )
        class_target = ClassModel(
            class_id="cls_target_m2",
            name="10-А",
            teacher_uuid="usr_owner_http",
            subject="tatar_language",
        )
        db_session.add_all([teacher_owner, teacher_other, class_target])
        await db_session.commit()

        token_other = create_access_token({"sub": "usr_other_http", "email": "other_http@tatar.ru", "role": "teacher"})
        res = await client.get("/api/v1/classes/cls_target_m2/students", headers={"Authorization": f"Bearer {token_other}"})
        assert res.status_code == 403
        assert res.json()["error_code"] == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_require_admin_guard(self, client: AsyncClient, db_session: AsyncSession):
        # Teacher token attempting admin dependency
        teacher_user = UserModel(id="usr_t_guard", email="t_guard@tatar.ru", full_name="Учитель Гард", status="active", role="teacher")
        admin_user = UserModel(id="usr_a_guard", email="a_guard@tatar.ru", full_name="Админ Гард", status="active", role="admin")
        db_session.add_all([teacher_user, admin_user])
        await db_session.commit()

        # Direct dependency test
        teacher_token = create_access_token({"sub": "usr_t_guard", "email": "t_guard@tatar.ru", "role": "teacher"})
        with pytest.raises(PermissionDeniedError) as exc_perm:
            require_admin(current_user=teacher_user)
        assert exc_perm.value.status_code == 403

        # Admin user allowed
        ad = require_admin(current_user=admin_user)
        assert ad.id == "usr_a_guard"

    @pytest.mark.asyncio
    async def test_require_superadmin_guard(self, client: AsyncClient):
        admin_user = UserModel(id="usr_admin_only", email="admin_only@tatar.ru", full_name="Только Админ", status="active", role="admin")
        superadmin_user = UserModel(id="usr_super", email="super@tatar.ru", full_name="Суперадмин", status="active", role="superadmin")

        with pytest.raises(PermissionDeniedError):
            require_superadmin(current_user=admin_user)

        s = require_superadmin(current_user=superadmin_user)
        assert s.role == "superadmin"

    @pytest.mark.asyncio
    async def test_get_owned_class_dependency(self, db_session: AsyncSession):
        owner = UserModel(id="usr_o1", email="o1@tatar.ru", full_name="Владелец О1", status="active", role="teacher")
        other = UserModel(id="usr_o2", email="o2@tatar.ru", full_name="Другой О2", status="active", role="teacher")
        cls_obj = ClassModel(class_id="cls_dep_test", name="11-А", teacher_uuid="usr_o1")
        db_session.add_all([owner, other, cls_obj])
        await db_session.commit()

        # Non-existent class raises NotFoundError (404)
        with pytest.raises(NotFoundError):
            await get_owned_class("cls_missing_999", current_user=owner, db=db_session)

        # Other teacher raises PermissionDeniedError (403)
        with pytest.raises(PermissionDeniedError):
            await get_owned_class("cls_dep_test", current_user=other, db=db_session)

        # Owner gets class
        res = await get_owned_class("cls_dep_test", current_user=owner, db=db_session)
        assert res.class_id == "cls_dep_test"
