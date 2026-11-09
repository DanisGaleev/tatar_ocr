import asyncio
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import IntegrityError

from app.core.database import Base
from app.models.user import UserModel
from app.models.school import ClassModel, StudentModel, ClassAssignmentModel
from app.models.submission import SubmissionModel
from app.models.test import AssembledTestModel
from app.models.audit import AuditLogModel
from app.models.session import UserSessionModel

from app.repositories import (
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

# Test in-memory engine with StaticPool
STRESS_DB_URL = "sqlite+aiosqlite:///:memory:"

stress_engine = create_async_engine(
    STRESS_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
StressSessionLocal = async_sessionmaker(
    bind=stress_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_test_schema():
    async with stress_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with stress_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def session():
    async with StressSessionLocal() as sess:
        yield sess
        await sess.rollback()


# =====================================================================
# 1. CONCURRENT WRITES AND READS ACROSS REPOSITORIES
# =====================================================================

@pytest.mark.asyncio
async def test_concurrent_writes_and_reads_all_repos():
    """
    Stress-test concurrent writes and reads across UserRepository, ClassRepository,
    StudentRepository, SubmissionRepository, and AuditRepository.
    Spawns 20 parallel async worker tasks each hitting all 5 repositories.
    """
    async def worker(worker_id: int):
        async with StressSessionLocal() as worker_sess:
            user_repo = UserRepository(worker_sess)
            class_repo = ClassRepository(worker_sess)
            student_repo = StudentRepository(worker_sess)
            submission_repo = SubmissionRepository(worker_sess)
            audit_repo = AuditRepository(worker_sess)

            # 1. Create unique user
            user = await user_repo.create_user(
                email=f"concurrent_{worker_id}_{uuid.uuid4().hex[:6]}@school.ru",
                full_name=f"Учитель {worker_id}",
                school_name=f"Гимназия №{worker_id}",
                role="teacher",
                auto_commit=True,
            )
            assert user.id is not None

            # 2. Create class
            class_id = f"cls_conc_{worker_id}_{uuid.uuid4().hex[:6]}"
            cls_obj = await class_repo.create_class(
                class_id=class_id,
                name=f"{worker_id}-А сыйныфы",
                teacher_uuid=user.id,
                auto_commit=True,
            )
            assert cls_obj.class_id == class_id

            # 3. Create students (1 individual, 2 bulk)
            s1 = await student_repo.create_student(
                class_id=class_id,
                last_name=f"Иванов_{worker_id}",
                first_name="Айрат",
                auto_commit=True,
            )
            bulk_students = await student_repo.bulk_create(
                [
                    {"class_id": class_id, "last_name": f"Закиров_{worker_id}", "first_name": "Булат"},
                    {"class_id": class_id, "last_name": f"Хәбибуллин_{worker_id}", "first_name": "Динар"},
                ],
                auto_commit=True,
            )
            assert len(bulk_students) == 2

            # 4. Create submission
            client_sub_uuid = str(uuid.uuid4())
            sub, is_new = await submission_repo.save_or_update(
                submission_data={
                    "client_submission_uuid": client_sub_uuid,
                    "assignment_id": f"asg_conc_{worker_id}",
                    "class_id": class_id,
                    "student_id": s1.student_id,
                    "student_name": s1.full_name,
                    "overall_score": 7.5,
                    "max_score": 8.0,
                    "final_grade": 5,
                    "teacher_id": user.id,
                    "questions_results": [{"q": 1, "score": 1.0}],
                },
                auto_commit=True,
            )
            assert is_new is True

            # 5. Log audit event
            log_entry = await audit_repo.log_event(
                action="CONCURRENT_TEST_OP",
                user_id=user.id,
                details={"worker": worker_id, "sub_id": sub.submission_id},
                auto_commit=True,
            )
            assert log_entry.id is not None

            # 6. Read back and verify cross-consistency
            u_fetched = await user_repo.get_by_email(user.email)
            assert u_fetched is not None
            assert u_fetched.id == user.id

            cls_fetched = await class_repo.get_by_id_and_teacher(class_id, user.id)
            assert cls_fetched is not None

            students_in_cls = await student_repo.list_by_class(class_id)
            assert len(students_in_cls) == 3

            subs_in_cls = await submission_repo.list_by_class(class_id)
            assert len(subs_in_cls) == 1
            assert subs_in_cls[0].submission_id == sub.submission_id

            logs, count = await audit_repo.query_logs(action="CONCURRENT_TEST_OP", user_id=user.id)
            assert count >= 1

            return user.id

    tasks = [worker(i) for i in range(20)]
    user_ids = await asyncio.gather(*tasks)
    assert len(user_ids) == 20
    assert len(set(user_ids)) == 20


@pytest.mark.asyncio
async def test_concurrent_updates_to_single_user_and_idempotency():
    """
    Test concurrent update operations on the same entity.
    Multiple workers attempt to increment failed attempts and update last login.
    """
    async with StressSessionLocal() as init_sess:
        u_repo = UserRepository(init_sess)
        shared_user = await u_repo.create_user(
            email=f"target_{uuid.uuid4().hex[:6]}@contention.ru",
            full_name="Target Under Contention",
            auto_commit=True,
        )
        target_id = shared_user.id

    async def update_worker(w_id: int):
        async with StressSessionLocal() as w_sess:
            repo = UserRepository(w_sess)
            u = await repo.increment_failed_attempts(target_id, max_attempts=10, auto_commit=True)
            await repo.update_last_login(u, auto_commit=True)
            return u.failed_login_attempts

    tasks = [update_worker(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 10

    async with StressSessionLocal() as check_sess:
        repo = UserRepository(check_sess)
        final_user = await repo.get_by_id(target_id)
        assert final_user is not None
        assert final_user.failed_login_attempts >= 1
        assert final_user.last_login_at is not None


# =====================================================================
# 2. ROLLBACK ATOMICITY
# =====================================================================

@pytest.mark.asyncio
async def test_rollback_atomicity_multi_repo_leaves_zero_traces():
    """
    Verify that when an operation involves multiple repositories with auto_commit=False,
    and a mid-way failure occurs, a subsequent rollback leaves absolutely NO partial records.
    """
    marker = uuid.uuid4().hex[:8]
    test_email = f"atomic_{marker}@test.com"
    test_class_id = f"cls_atomic_{marker}"
    test_stu_id = f"stu_atomic_{marker}"
    test_sub_client_uuid = str(uuid.uuid4())

    async with StressSessionLocal() as atomic_sess:
        user_repo = UserRepository(atomic_sess)
        class_repo = ClassRepository(atomic_sess)
        student_repo = StudentRepository(atomic_sess)
        submission_repo = SubmissionRepository(atomic_sess)
        audit_repo = AuditRepository(atomic_sess)

        # 1. Stage user
        u = await user_repo.create_user(
            email=test_email,
            full_name="Atomic Tester",
            auto_commit=False,
        )
        # 2. Stage class
        cls_obj = await class_repo.create_class(
            class_id=test_class_id,
            name="Atomic Class",
            teacher_uuid=u.id,
            auto_commit=False,
        )
        # 3. Stage student
        stu = await student_repo.create_student(
            class_id=test_class_id,
            student_id=test_stu_id,
            last_name="Сәйфи",
            first_name="Фәрит",
            auto_commit=False,
        )
        # 4. Stage submission
        await submission_repo.save_or_update(
            submission_data={
                "client_submission_uuid": test_sub_client_uuid,
                "assignment_id": "asg_atomic",
                "class_id": test_class_id,
                "student_id": test_stu_id,
                "student_name": "Сәйфи Фәрит",
                "overall_score": 6.0,
            },
            auto_commit=False,
        )
        # 5. Stage audit log
        await audit_repo.log_event(
            action="ATOMIC_STAGE",
            user_id=u.id,
            auto_commit=False,
        )

        # Deliberately raise simulated business failure and rollback
        await atomic_sess.rollback()

    # Now verify in a fresh independent session: NOTHING must exist
    async with StressSessionLocal() as verify_sess:
        v_user_repo = UserRepository(verify_sess)
        v_class_repo = ClassRepository(verify_sess)
        v_student_repo = StudentRepository(verify_sess)
        v_sub_repo = SubmissionRepository(verify_sess)
        v_audit_repo = AuditRepository(verify_sess)

        assert await v_user_repo.get_by_email(test_email) is None
        assert await v_class_repo.get_by_id(test_class_id) is None
        assert await v_student_repo.get_by_id(test_stu_id) is None
        assert await v_sub_repo.get_by_client_uuid(test_sub_client_uuid) is None
        logs, count = await v_audit_repo.query_logs(action="ATOMIC_STAGE")
        assert count == 0


@pytest.mark.asyncio
async def test_rollback_on_integrity_error_cleans_uncommitted_state():
    """
    Verify that an unhandled IntegrityError rolled back leaves no partially flushed objects.
    """
    marker = uuid.uuid4().hex[:8]
    valid_email = f"valid_{marker}@test.com"

    async with StressSessionLocal() as err_sess:
        repo = UserRepository(err_sess)
        # Add first valid user with auto_commit=False
        u1 = await repo.create_user(
            email=valid_email,
            full_name="Valid First User",
            auto_commit=False,
        )
        # Intentionally force duplicate email insertion violating UNIQUE constraint
        with pytest.raises(IntegrityError):
            u_duplicate = UserModel(
                id=f"usr_{uuid.uuid4().hex[:8]}",
                email=valid_email,
                full_name="Duplicate Violator",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            err_sess.add(u_duplicate)
            await err_sess.flush()

        # Rollback the transaction after error
        await err_sess.rollback()

    # Verify valid_email was NOT saved
    async with StressSessionLocal() as check_sess:
        repo = UserRepository(check_sess)
        assert await repo.get_by_email(valid_email) is None


# =====================================================================
# 3. CASCADE DELETE OPERATIONS
# =====================================================================

@pytest.mark.asyncio
async def test_cascade_delete_class_cleans_all_dependents_and_preserves_others():
    """
    Verify that ClassRepository.delete_cascade:
    1. Removes the class
    2. Removes all related students
    3. Removes all related assignments
    4. Removes all related submissions
    5. Leaves shared test definitions intact
    6. Leaves other classes and their data completely untouched.
    """
    prefix = uuid.uuid4().hex[:6]
    c1_id = f"c1_{prefix}"
    c2_id = f"c2_{prefix}"
    teacher_id = f"t_{prefix}"

    async with StressSessionLocal() as sess:
        class_repo = ClassRepository(sess)
        student_repo = StudentRepository(sess)
        asg_repo = AssignmentRepository(sess)
        sub_repo = SubmissionRepository(sess)

        # Create shared test definition
        test_def = AssembledTestModel(
            test_id=f"test_{prefix}",
            title="Shared Tatar Grammar Test",
            bundle_json="{}",
        )
        sess.add(test_def)
        await sess.commit()

        # Setup Class 1 (Target for cascade deletion)
        await class_repo.create_class(class_id=c1_id, name="Class 1", teacher_uuid=teacher_id)
        s1 = await student_repo.create_student(class_id=c1_id, last_name="Абдуллин", first_name="Карим")
        s2 = await student_repo.create_student(class_id=c1_id, last_name="Бәширов", first_name="Ильяс")
        await asg_repo.create_assignment(class_id=c1_id, assignment_id=test_def.test_id)
        await sub_repo.save_or_update(
            submission_data={
                "client_submission_uuid": f"sub_c1_{prefix}_1",
                "assignment_id": test_def.test_id,
                "class_id": c1_id,
                "student_id": s1.student_id,
                "student_name": s1.full_name,
            }
        )

        # Setup Class 2 (Should remain completely untouched)
        await class_repo.create_class(class_id=c2_id, name="Class 2", teacher_uuid=teacher_id)
        s3 = await student_repo.create_student(class_id=c2_id, last_name="Вәлиева", first_name="Ләйсән")
        await asg_repo.create_assignment(class_id=c2_id, assignment_id=test_def.test_id)
        await sub_repo.save_or_update(
            submission_data={
                "client_submission_uuid": f"sub_c2_{prefix}_1",
                "assignment_id": test_def.test_id,
                "class_id": c2_id,
                "student_id": s3.student_id,
                "student_name": s3.full_name,
            }
        )

    # Perform cascade deletion of Class 1
    async with StressSessionLocal() as del_sess:
        del_class_repo = ClassRepository(del_sess)
        deleted = await del_class_repo.delete_cascade(c1_id)
        assert deleted is True

    # Assertions on post-deletion state
    async with StressSessionLocal() as check_sess:
        c_repo = ClassRepository(check_sess)
        stu_repo = StudentRepository(check_sess)
        a_repo = AssignmentRepository(check_sess)
        s_repo = SubmissionRepository(check_sess)

        # C1 must be completely gone
        assert await c_repo.get_by_id(c1_id) is None
        assert len(await stu_repo.list_by_class(c1_id, active_only=False)) == 0
        assert len(await a_repo.list_by_class(c1_id)) == 0
        assert len(await s_repo.list_by_class(c1_id)) == 0

        # Shared test definition remains intact
        assert await a_repo.get_by_id(test_def.test_id) is not None

        # C2 must remain completely intact
        assert await c_repo.get_by_id(c2_id) is not None
        c2_students = await stu_repo.list_by_class(c2_id)
        assert len(c2_students) == 1
        assert c2_students[0].student_id == s3.student_id
        assert len(await a_repo.list_by_class(c2_id)) == 1
        assert len(await s_repo.list_by_class(c2_id)) == 1

        # Non-existent delete returns False gracefully
        assert await c_repo.delete_cascade("ghost_class_id") is False


@pytest.mark.asyncio
async def test_teacher_data_purge_cascades_all_classes():
    """
    Verify that ClassRepository.delete_all_by_teacher purges all classes for that teacher
    while leaving other teachers' classes unaffected.
    """
    t1 = f"t1_{uuid.uuid4().hex[:6]}"
    t2 = f"t2_{uuid.uuid4().hex[:6]}"

    async with StressSessionLocal() as sess:
        repo = ClassRepository(sess)
        c1 = await repo.create_class(class_id=f"c1_{t1}", name="T1 Class 1", teacher_uuid=t1)
        c2 = await repo.create_class(class_id=f"c2_{t1}", name="T1 Class 2", teacher_uuid=t1)
        c3 = await repo.create_class(class_id=f"c3_{t2}", name="T2 Class 1", teacher_uuid=t2)

    async with StressSessionLocal() as purge_sess:
        p_repo = ClassRepository(purge_sess)
        count = await p_repo.delete_all_by_teacher(t1)
        assert count == 2

    async with StressSessionLocal() as verify_sess:
        v_repo = ClassRepository(verify_sess)
        assert len(await v_repo.list_by_teacher(t1, include_unassigned=False)) == 0
        t2_classes = await v_repo.list_by_teacher(t2, include_unassigned=False)
        assert len(t2_classes) == 1
        assert t2_classes[0].class_id == f"c3_{t2}"


# =====================================================================
# 4. EMPTY AND BOUNDARY QUERIES & CYRILLIC SEARCH
# =====================================================================

@pytest.mark.asyncio
async def test_pagination_boundary_cases(session: AsyncSession):
    """
    Stress-test pagination with negative, zero, and huge out-of-bounds values.
    """
    user_repo = UserRepository(session)
    audit_repo = AuditRepository(session)

    # 1. Out of bounds page
    users, total = await user_repo.list_users(page=999999, page_size=20)
    assert users == []
    assert total >= 0

    # 2. Zero page (skip = max(0, (0-1)*20) -> skip=0)
    users_p0, total_p0 = await user_repo.list_users(page=0, page_size=10)
    assert total_p0 == total
    assert len(users_p0) <= 10

    # 3. Negative page
    users_neg, total_neg = await user_repo.list_users(page=-5, page_size=10)
    assert total_neg == total
    assert len(users_neg) <= 10

    # 4. Zero page size in audit repo
    logs, log_total = await audit_repo.query_logs(page=1, page_size=0)
    assert logs == [] or len(logs) == 0

    # 5. Out of bounds audit page
    logs_huge, _ = await audit_repo.query_logs(page=500000, page_size=100)
    assert logs_huge == []


@pytest.mark.asyncio
async def test_non_existent_entity_lookups_return_none(session: AsyncSession):
    """
    Ensure lookups for non-existent identifiers return None / empty without throwing exceptions.
    """
    user_repo = UserRepository(session)
    class_repo = ClassRepository(session)
    student_repo = StudentRepository(session)
    submission_repo = SubmissionRepository(session)
    asg_repo = AssignmentRepository(session)
    session_repo = SessionRepository(session)
    audit_repo = AuditRepository(session)

    assert await user_repo.get_by_id("non_existent_id") is None
    assert await user_repo.get_by_email("ghost_user@nowhere.tld") is None
    assert await class_repo.get_by_id("non_existent_class") is None
    assert await class_repo.get_by_id_and_teacher("non_existent_class", "ghost_teacher") is None
    assert await student_repo.get_by_id("non_existent_student") is None
    assert await student_repo.get_by_id("stu_123", class_id="non_existent_class") is None
    assert await submission_repo.get_by_id("non_existent_sub") is None
    assert await submission_repo.get_by_client_uuid("00000000-0000-0000-0000-000000000000") is None
    assert await asg_repo.get_by_id("non_existent_test") is None
    assert await asg_repo.get_class_assignment("non_existent_cls", "non_existent_asg") is None
    assert await session_repo.get_by_token_hash("ghost_hash") is None
    assert await session_repo.get_active_by_token_hash("ghost_hash") is None

    # Empty list checks
    assert await class_repo.list_by_teacher("ghost_teacher", include_unassigned=False) == []
    assert await student_repo.list_by_class("ghost_class") == []
    assert await submission_repo.list_by_class("ghost_class") == []
    assert await submission_repo.list_by_student("ghost_student") == []
    logs, total_logs = await audit_repo.query_logs(action="DEFINITELY_NO_SUCH_ACTION_RECORDED")
    assert logs == []
    assert total_logs == 0


@pytest.mark.asyncio
async def test_search_special_characters_sql_wildcards_and_cyrillic(session: AsyncSession):
    """
    Adversarial test on UserRepository full text search:
    - Cyrillic letters with Tatar alphabet specifics (Ә, Ө, Ү, Җ, Ң, Һ).
    - Case sensitivity folding (e.g. searching 'әминә' matches 'Әминә').
    - SQL wildcard characters (% and _).
    - SQL injection pattern strings.
    - Quotes, regex symbols, and backslashes.
    """
    user_repo = UserRepository(session)

    # Seed diverse users
    u_amina = await user_repo.create_user(
        email=f"amina_{uuid.uuid4().hex[:6]}@tatar.ru",
        full_name="Әминә Гайфуллина",
        school_name="12нче гимназия 'Өмет'",
        auto_commit=True,
    )
    u_jalil = await user_repo.create_user(
        email=f"jalil_{uuid.uuid4().hex[:6]}@tatar.ru",
        full_name="Муса Җәлил",
        school_name="Казан 1нче мәктәбе",
        auto_commit=True,
    )
    u_special = await user_repo.create_user(
        email=f"percent_{uuid.uuid4().hex[:6]}@domain.org",
        full_name="Mr. 100% Score_Master",
        school_name="Formula & Tech [O'Reilly]",
        auto_commit=True,
    )

    # 1. Tatar Cyrillic search (lowercase search vs titlecase stored name)
    found_amina, count_amina = await user_repo.list_users(search="әминә")
    assert count_amina >= 1
    assert any(u.id == u_amina.id for u in found_amina)

    # 2. Tatar Cyrillic search with unique letter Җ
    found_jalil, count_jalil = await user_repo.list_users(search="җәлил")
    assert count_jalil >= 1
    assert any(u.id == u_jalil.id for u in found_jalil)

    # 3. Tatar specific letter in school name (Өмет)
    found_omet, count_omet = await user_repo.list_users(search="өмет")
    assert count_omet >= 1
    assert any(u.id == u_amina.id for u in found_omet)

    # 4. Search with single quote
    found_quote, _ = await user_repo.list_users(search="O'Reilly")
    assert any(u.id == u_special.id for u in found_quote)

    # 5. Search with percent sign '%'
    found_percent, _ = await user_repo.list_users(search="100%")
    assert any(u.id == u_special.id for u in found_percent)

    # 6. Search with underscore '_'
    found_under, _ = await user_repo.list_users(search="Score_Master")
    assert any(u.id == u_special.id for u in found_under)

    # 7. Search with regex symbols: . * + [ ] ( )
    found_regex, _ = await user_repo.list_users(search="[O'Reilly]")
    assert any(u.id == u_special.id for u in found_regex)

    # 8. SQL injection attempt string should be treated as literal and not error
    sql_payload = "'; DROP TABLE users; --"
    found_inj, count_inj = await user_repo.list_users(search=sql_payload)
    assert found_inj == []
    assert count_inj == 0

    # 9. Whitespace-only search returns standard unfiltered listing without error
    users_blank, total_blank = await user_repo.list_users(search="    ")
    assert total_blank >= 3


# =====================================================================
# 5. DATABASE LOCK / SQLITE STATIC POOL STRESS (50+ OPS)
# =====================================================================

@pytest.mark.asyncio
async def test_static_pool_stress_50_plus_concurrent_operations():
    """
    Adversarial lock test on in-memory StaticPool:
    Spawns 60 concurrent worker tasks executing simultaneous writes, reads,
    and commits to verify no 'OperationalError: database is locked' occurs.
    """
    async def stress_worker(worker_idx: int):
        async with StressSessionLocal() as w_sess:
            u_repo = UserRepository(w_sess)
            c_repo = ClassRepository(w_sess)
            s_repo = StudentRepository(w_sess)
            a_repo = AuditRepository(w_sess)

            # Concurrent user creation
            u = await u_repo.create_user(
                email=f"pool_stress_{worker_idx}_{uuid.uuid4().hex[:6]}@tatar.ru",
                full_name=f"Стресс Тестер {worker_idx}",
                auto_commit=True,
            )

            # Concurrent audit log
            await a_repo.log_event(
                action="POOL_STRESS_OP",
                user_id=u.id,
                details={"idx": worker_idx},
                auto_commit=True,
            )

            # Concurrent class and student
            c = await c_repo.create_class(
                class_id=f"cls_stress_{worker_idx}_{uuid.uuid4().hex[:6]}",
                name=f"Стресс Класс {worker_idx}",
                teacher_uuid=u.id,
                auto_commit=True,
            )
            await s_repo.create_student(
                class_id=c.class_id,
                last_name=f"Студент_{worker_idx}",
                first_name="Татар",
                auto_commit=True,
            )

            # Concurrent read query
            _, count = await u_repo.list_users(page=1, page_size=5)
            assert count >= 1
            return u.id

    tasks = [stress_worker(i) for i in range(60)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 60


@pytest.mark.asyncio
async def test_file_based_sqlite_stress_50_plus_concurrent_operations():
    """
    Adversarial lock test on a real file-based SQLite database:
    Spawns 50+ concurrent worker tasks executing write transactions simultaneously
    to ensure database locking / busy-timeout does not raise OperationalError.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_file = tmp.name

    file_engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_file}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    FileSessionLocal = async_sessionmaker(
        bind=file_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    try:
        # Create schema in the temporary file DB
        async with file_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async def file_worker(idx: int):
            async with FileSessionLocal() as f_sess:
                u_repo = UserRepository(f_sess)
                a_repo = AuditRepository(f_sess)
                c_repo = ClassRepository(f_sess)

                # Write user
                user = await u_repo.create_user(
                    email=f"file_stress_{idx}_{uuid.uuid4().hex[:6]}@tatar.ru",
                    full_name=f"Файл Тестер {idx}",
                    auto_commit=True,
                )
                # Write audit
                await a_repo.log_event(
                    action="FILE_STRESS_LOG",
                    user_id=user.id,
                    auto_commit=True,
                )
                # Write class
                await c_repo.create_class(
                    class_id=f"fcls_{idx}_{uuid.uuid4().hex[:6]}",
                    name=f"Класс {idx}",
                    teacher_uuid=user.id,
                    auto_commit=True,
                )
                # Read query
                users, total = await u_repo.list_users(page=1, page_size=10)
                assert total >= 1
                return user.id

        tasks = [file_worker(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 50

    finally:
        await file_engine.dispose()
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except Exception:
                pass


# =====================================================================
# 6. STUDENT NAME NORMALIZATION & SUBMISSION IDEMPOTENCY
# =====================================================================

@pytest.mark.asyncio
async def test_student_name_formatting_and_normalization(session: AsyncSession):
    """
    Adversarial verification of name normalization in StudentRepository:
    Trimming, middle name presence/absence, short name generation with Tatar letters.
    """
    student_repo = StudentRepository(session)

    # 1. Names with leading/trailing whitespaces
    s1 = await student_repo.create_student(
        class_id="norm_class",
        last_name="   Хәйруллина   ",
        first_name="   Гөлназ   ",
        middle_name="   Рамилевна   ",
    )
    assert s1.last_name == "Хәйруллина"
    assert s1.first_name == "Гөлназ"
    assert s1.middle_name == "Рамилевна"
    assert s1.full_name == "Хәйруллина Гөлназ Рамилевна"
    assert s1.short_name == "Хәйруллина Г."

    # 2. Student without middle name
    s2 = await student_repo.create_student(
        class_id="norm_class",
        last_name="Сабиров",
        first_name="Тимур",
        middle_name="",
    )
    assert s2.full_name == "Сабиров Тимур"
    assert s2.short_name == "Сабиров Т."

    # 3. Custom short name override
    s3 = await student_repo.create_student(
        class_id="norm_class",
        last_name="Галиев",
        first_name="Ильдар",
        short_name="И. Галиев (ст)",
    )
    assert s3.short_name == "И. Галиев (ст)"


@pytest.mark.asyncio
async def test_submission_batch_upsert_idempotency_and_duplicates(session: AsyncSession):
    """
    Test SubmissionRepository.batch_create idempotency:
    Verifies that duplicate client_submission_uuid in a batch updates rather than errors.
    """
    sub_repo = SubmissionRepository(session)
    shared_client_uuid = str(uuid.uuid4())

    batch_payload = [
        {
            "client_submission_uuid": shared_client_uuid,
            "assignment_id": "asg_idemp",
            "class_id": "cls_idemp",
            "student_id": "stu_idemp_1",
            "overall_score": 5.0,
            "final_grade": 3,
        },
        {
            "client_submission_uuid": shared_client_uuid,  # Same UUID updated in same batch
            "assignment_id": "asg_idemp",
            "class_id": "cls_idemp",
            "student_id": "stu_idemp_1",
            "overall_score": 8.0,
            "final_grade": 5,
        },
        {
            "client_submission_uuid": str(uuid.uuid4()),  # Distinct submission
            "assignment_id": "asg_idemp",
            "class_id": "cls_idemp",
            "student_id": "stu_idemp_2",
            "overall_score": 7.0,
            "final_grade": 4,
        },
    ]

    inserted, updated = await sub_repo.batch_create(batch_payload, auto_commit=True)
    assert inserted == 2
    assert updated == 1

    # Verify final state for shared_client_uuid is updated to 8.0 score
    final_sub = await sub_repo.get_by_client_uuid(shared_client_uuid)
    assert final_sub is not None
    assert final_sub.overall_score == 8.0
    assert final_sub.final_grade == 5


# =====================================================================
# STANDALONE VERIFICATION RUNNER
# =====================================================================

async def run_standalone_verification():
    print("=====================================================================")
    print("STARTING STANDALONE ADVERSARIAL REPOSITORY VERIFICATION HARNESS")
    print("=====================================================================")

    async with stress_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    tests = [
        ("test_concurrent_writes_and_reads_all_repos", test_concurrent_writes_and_reads_all_repos),
        ("test_concurrent_updates_to_single_user_and_idempotency", test_concurrent_updates_to_single_user_and_idempotency),
        ("test_rollback_atomicity_multi_repo_leaves_zero_traces", test_rollback_atomicity_multi_repo_leaves_zero_traces),
        ("test_rollback_on_integrity_error_cleans_uncommitted_state", test_rollback_on_integrity_error_cleans_uncommitted_state),
        ("test_cascade_delete_class_cleans_all_dependents_and_preserves_others", test_cascade_delete_class_cleans_all_dependents_and_preserves_others),
        ("test_teacher_data_purge_cascades_all_classes", test_teacher_data_purge_cascades_all_classes),
        ("test_static_pool_stress_50_plus_concurrent_operations", test_static_pool_stress_50_plus_concurrent_operations),
        ("test_file_based_sqlite_stress_50_plus_concurrent_operations", test_file_based_sqlite_stress_50_plus_concurrent_operations),
    ]

    session_tests = [
        ("test_pagination_boundary_cases", test_pagination_boundary_cases),
        ("test_non_existent_entity_lookups_return_none", test_non_existent_entity_lookups_return_none),
        ("test_search_special_characters_sql_wildcards_and_cyrillic", test_search_special_characters_sql_wildcards_and_cyrillic),
        ("test_student_name_formatting_and_normalization", test_student_name_formatting_and_normalization),
        ("test_submission_batch_upsert_idempotency_and_duplicates", test_submission_batch_upsert_idempotency_and_duplicates),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            start_t = datetime.now()
            await test_fn()
            dur = (datetime.now() - start_t).total_seconds()
            print(f"  [PASS] {name} ({dur:.3f}s)")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    for name, test_fn in session_tests:
        async with StressSessionLocal() as sess:
            try:
                start_t = datetime.now()
                await test_fn(sess)
                dur = (datetime.now() - start_t).total_seconds()
                print(f"  [PASS] {name} ({dur:.3f}s)")
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                failed += 1

    async with stress_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await stress_engine.dispose()

    print("=====================================================================")
    print(f"VERIFICATION SUMMARY: {passed} PASSED, {failed} FAILED")
    print("=====================================================================")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(run_standalone_verification())

