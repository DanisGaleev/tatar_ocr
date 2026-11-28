import io
import json
import openpyxl
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

from app.main import app, seed_initial_data
from app.core.database import Base, get_db, _set_sqlite_pragma
from app.models.school import ClassModel, StudentModel
from app.models.teacher import TeacherModel
from app.models.test import AssembledTestModel

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def override_get_db():
    async with TestSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@pytest_asyncio.fixture(autouse=True)
async def prepare_database():
    app.dependency_overrides[get_db] = override_get_db
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        teacher = TeacherModel(
            teacher_uuid="teacher-custom-scale",
            device_os="android",
            app_version="1.0.4",
            teacher_name="Test Teacher",
            school_name="School 1",
            grade_5_min_pct=90,
            grade_4_min_pct=75,
            grade_3_min_pct=60,
            confidence_flag_threshold=0.65,
        )
        cls = ClassModel(
            class_id="cls_7a_test",
            teacher_uuid="teacher-custom-scale",
            name="7/А",
            subject="tatar_language",
            academic_year="2026-2027",
        )
        s1 = StudentModel(
            student_id="stu_a1",
            class_id="cls_7a_test",
            last_name="=SUM(A1)",
            first_name="Амир",
            middle_name="",
            full_name="=SUM(A1) Амир",
            short_name="=SUM(A1) А.",
        )
        s2 = StudentModel(
            student_id="stu_a2",
            class_id="cls_7a_test",
            last_name="Закирова",
            first_name="Дилә",
            middle_name="Нияз кызы",
            full_name="Закирова Дилә Нияз кызы",
            short_name="Закирова Д.",
        )
        asm = AssembledTestModel(
            test_id="TEST-MV-01",
            title="Мультивариант тест",
            grade_level=7,
            total_variants=2,
            bundle_json=json.dumps(
                {
                    "assignment_id": "TEST-MV-01",
                    "title": "Мультивариант тест",
                    "total_variants": 2,
                    "variants": [
                        {
                            "variant_id": 1,
                            "questions": [
                                {"question_number": 1, "marker_id": 11, "prompt": "V1 Q1 китап ->", "topic_tag": "case_dative", "cell_count": 8, "expected_answer": "КИТАПКА"},
                                {"question_number": 2, "marker_id": 12, "prompt": "V1 Q2 бала ->", "topic_tag": "plural_affixes", "cell_count": 8, "expected_answer": "БАЛАЛАР"},
                            ],
                        },
                        {
                            "variant_id": 2,
                            "questions": [
                                {"question_number": 1, "marker_id": 11, "prompt": "V2 Q1 өстәл ->", "topic_tag": "case_ablative", "cell_count": 8, "expected_answer": "ӨСТӘЛДӘН"},
                                {"question_number": 2, "marker_id": 12, "prompt": "V2 Q2 урман ->", "topic_tag": "plural_affixes", "cell_count": 8, "expected_answer": "УРМАННАР"},
                            ],
                        },
                    ],
                },
                ensure_ascii=False,
            ),
        )
        session.add_all([teacher, cls, s1, s2, asm])
        await session.commit()

    yield

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestStage1CriticalFixes:
    @pytest.mark.asyncio
    async def test_ac1_exception_handler_no_nameerror_and_sqlite_fk_pragma(self):
        from starlette.requests import Request
        from app.main import unhandled_exception_handler

        req = Request({"type": "http", "method": "GET", "path": "/boom", "headers": []})
        resp = await unhandled_exception_handler(req, RuntimeError("boom"))
        assert resp.status_code == 500
        assert json.loads(resp.body)["detail"] == "Internal server error"

        # Verify SQLite FK pragma helper and CORS middleware configuration
        import sqlite3
        raw_conn = sqlite3.connect(":memory:")
        _set_sqlite_pragma(raw_conn, None)
        fk_state = raw_conn.execute("PRAGMA foreign_keys").fetchone()[0]
        raw_conn.close()
        assert fk_state == 1

        cors = next(m for m in app.user_middleware if "CORSMiddleware" in str(m.cls))
        assert cors.kwargs["allow_credentials"] is False

    @pytest.mark.asyncio
    async def test_ac1_seed_does_not_overwrite_existing_test(self, monkeypatch):
        from app import main as main_mod

        monkeypatch.setattr(main_mod, "AsyncSessionLocal", TestSessionLocal)
        async with TestSessionLocal() as session:
            custom = AssembledTestModel(
                test_id="TAT-2026-Q1",
                title="Modified Title",
                grade_level=7,
                total_variants=1,
                bundle_json='{"custom": true}',
            )
            session.add(custom)
            await session.commit()

        await seed_initial_data()
        async with TestSessionLocal() as session:
            row = await session.get(AssembledTestModel, "TAT-2026-Q1")
            assert row is not None
            assert row.bundle_json == '{"custom": true}'

    @pytest.mark.asyncio
    async def test_ac2_verify_answer_rejects_overflow_zero_and_extra_chars(self, client: AsyncClient):
        r_zero = await client.post("/api/v1/constructor/verify-answer", json={
            "expected_answer": "АК",
            "student_answer": "АК",
            "cell_count": 0,
        })
        assert r_zero.status_code == 422

        r_overflow = await client.post("/api/v1/constructor/verify-answer", json={
            "expected_answer": "СӘЛАМӘТЛЕК",
            "student_answer": "СӘЛАМӘТЛЕК",
            "cell_count": 8,
        })
        assert r_overflow.status_code == 422

        r_extra = await client.post("/api/v1/constructor/verify-answer", json={
            "expected_answer": "АК",
            "student_answer": "АКХ",
            "cell_count": 2,
        })
        assert r_extra.status_code == 200
        assert r_extra.json()["is_correct"] is False

    @pytest.mark.asyncio
    async def test_ac3_submission_validation_recalculation_and_dedup(self, client: AsyncClient):
        # Reject out-of-bounds score, invalid status, unknown student, unknown variant
        bad_score = await client.post("/api/v1/submissions", json={
            "assignment_id": "TEST-MV-01",
            "class_id": "cls_7a_test",
            "submission": {
                "student_id": "stu_a1",
                "student_name": "=SUM(A1) Амир",
                "variant": 1,
                "overall_score": 10.0,
                "max_score": 8.0,
                "final_grade": 5,
                "questions_results": [],
            },
        })
        assert bad_score.status_code == 422

        bad_student = await client.post("/api/v1/submissions", json={
            "assignment_id": "TEST-MV-01",
            "class_id": "cls_7a_test",
            "submission": {
                "student_id": "stu_unknown",
                "student_name": "Unknown",
                "variant": 1,
                "overall_score": 1.0,
                "max_score": 2.0,
                "final_grade": 2,
                "questions_results": [],
            },
        })
        assert bad_student.status_code == 422

        # Server recalculates tampered score/grade & respects teacher_override="CORRECT" + TeacherModel scale (90/75/60)
        tampered = await client.post("/api/v1/submissions", json={
            "assignment_id": "TEST-MV-01",
            "class_id": "cls_7a_test",
            "submission": {
                "client_submission_uuid": "uuid-stu-a1",
                "student_id": "stu_a1",
                "student_name": "Spoofed Name",
                "variant": 1,
                "overall_score": 0.0,
                "max_score": 8.0,
                "final_grade": 2,
                "questions_results": [
                    {
                        "question_number": 1,
                        "marker_id": 11,
                        "topic_tag": "case_dative",
                        "is_correct": False,
                        "points_earned": 0.0,
                        "cells": [
                            {"cell_index": i, "expected_char": c, "predicted_char": ("О" if i == 0 else c), "confidence": 0.90, "status": "FLAG_OVERRIDDEN_BY_TEACHER" if i == 0 else "MATCH", "teacher_override": "CORRECT" if i == 0 else None}
                            for i, c in enumerate("КИТАПКА")
                        ],
                    },
                    {
                        "question_number": 2,
                        "marker_id": 12,
                        "topic_tag": "plural_affixes",
                        "is_correct": False,
                        "points_earned": 0.0,
                        "cells": [
                            {"cell_index": i, "expected_char": c, "predicted_char": c, "confidence": 0.95, "status": "MATCH"}
                            for i, c in enumerate("БАЛАЛАР")
                        ],
                    },
                ],
            },
        })
        assert tampered.status_code == 200

        # Re-submitting same student with different client_submission_uuid updates in place (no duplicates)
        # Also duplicate UUID inside same batch does not crash with IntegrityError
        batch = await client.post("/api/v1/submissions/batch-sync", json={
            "assignment_id": "TEST-MV-01",
            "class_id": "cls_7a_test",
            "submissions": [
                {
                    "client_submission_uuid": "uuid-stu-a1-v2",
                    "student_id": "stu_a1",
                    "student_name": "Ignored",
                    "variant": 1,
                    "overall_score": 1.0,
                    "max_score": 2.0,
                    "final_grade": 2,
                    "questions_results": [
                        {
                            "question_number": 1,
                            "marker_id": 11,
                            "topic_tag": "case_dative",
                            "is_correct": True,
                            "points_earned": 1.0,
                            "cells": [{"cell_index": i, "expected_char": c, "predicted_char": ("О" if i == 0 else c), "confidence": 0.90, "status": "FLAG_OVERRIDDEN_BY_TEACHER" if i == 0 else "MATCH", "teacher_override": "CORRECT" if i == 0 else None} for i, c in enumerate("КИТАПКА")],
                        },
                        {
                            "question_number": 2,
                            "marker_id": 12,
                            "topic_tag": "plural_affixes",
                            "is_correct": True,
                            "points_earned": 1.0,
                            "cells": [{"cell_index": i, "expected_char": c, "predicted_char": c, "confidence": 0.95, "status": "MATCH"} for i, c in enumerate("БАЛАЛАР")],
                        },
                    ],
                },
                {
                    "client_submission_uuid": "uuid-stu-a2-dup",
                    "student_id": "stu_a2",
                    "student_name": "Закирова Дилә Нияз кызы",
                    "variant": 2,
                    "overall_score": 0.0,
                    "max_score": 2.0,
                    "final_grade": 2,
                    "questions_results": [
                        {
                            "question_number": 1,
                            "marker_id": 11,
                            "topic_tag": "case_ablative",
                            "is_correct": False,
                            "points_earned": 0.0,
                            "cells": [{"cell_index": i, "expected_char": c, "predicted_char": "А", "confidence": 0.90, "status": "MISMATCH"} for i, c in enumerate("ӨСТӘЛДӘН")],
                        },
                        {
                            "question_number": 2,
                            "marker_id": 12,
                            "topic_tag": "plural_affixes",
                            "is_correct": True,
                            "points_earned": 1.0,
                            "cells": [{"cell_index": i, "expected_char": c, "predicted_char": c, "confidence": 0.90, "status": "MATCH"} for i, c in enumerate("УРМАННАР")],
                        },
                    ],
                },
                {
                    "client_submission_uuid": "uuid-stu-a2-dup",
                    "student_id": "stu_a2",
                    "student_name": "Закирова Дилә Нияз кызы",
                    "variant": 2,
                    "overall_score": 1.0,
                    "max_score": 2.0,
                    "final_grade": 2,
                    "questions_results": [
                        {
                            "question_number": 1,
                            "marker_id": 11,
                            "topic_tag": "case_ablative",
                            "is_correct": False,
                            "points_earned": 0.0,
                            "cells": [{"cell_index": i, "expected_char": c, "predicted_char": "А", "confidence": 0.90, "status": "MISMATCH"} for i, c in enumerate("ӨСТӘЛДӘН")],
                        },
                        {
                            "question_number": 2,
                            "marker_id": 12,
                            "topic_tag": "plural_affixes",
                            "is_correct": True,
                            "points_earned": 1.0,
                            "cells": [{"cell_index": i, "expected_char": c, "predicted_char": c, "confidence": 0.90, "status": "MATCH"} for i, c in enumerate("УРМАННАР")],
                        },
                    ],
                },
            ],
        })
        assert batch.status_code == 200
        assert batch.json()["updated_count"] == 2
        assert batch.json()["inserted_count"] == 1

        # Cross-student UUID reuse must return 409 Conflict
        hijack = await client.post("/api/v1/submissions", json={
            "assignment_id": "TEST-MV-01",
            "class_id": "cls_7a_test",
            "submission": {
                "client_submission_uuid": "uuid-stu-a2-dup",
                "student_id": "stu_a1",
                "student_name": "A1",
                "variant": 1,
                "overall_score": 2.0,
                "max_score": 2.0,
                "final_grade": 5,
                "questions_results": [],
            },
        })
        assert hijack.status_code == 409

        # Check persisted recalculated scores: stu_a1 = 2/2 (grade 5), stu_a2 = 1/2 (50% < 60% -> grade 2)
        subs = (await client.get("/api/v1/submissions?class_id=cls_7a_test")).json()
        assert len(subs) == 2
        by_stu = {s["student_id"]: s for s in subs}
        assert by_stu["stu_a1"]["overall_score"] == 2.0
        assert by_stu["stu_a1"]["max_score"] == 2.0
        assert by_stu["stu_a1"]["final_grade"] == 5
        assert by_stu["stu_a1"]["student_name"] == "=SUM(A1) Амир"
        assert by_stu["stu_a2"]["overall_score"] == 1.0
        assert by_stu["stu_a2"]["final_grade"] == 2

    @pytest.mark.asyncio
    async def test_ac4_analytics_overrides_variants_empty_and_404(self, client: AsyncClient):
        # 404 for nonexistent entities
        assert (await client.get("/api/v1/analytics/students/missing")).status_code == 404
        assert (await client.get("/api/v1/analytics/classes/missing")).status_code == 404
        assert (await client.get("/api/v1/analytics/assignments/missing")).status_code == 404

        # Zero-submission student returns NO_DATA / None
        empty_stu = await client.get("/api/v1/analytics/students/stu_a1")
        assert empty_stu.status_code == 200
        assert empty_stu.json()["handwriting_quality_pct"] is None
        assert empty_stu.json()["handwriting_status"] == "NO_DATA"

        # Submit Variant 1 (teacher_override=CORRECT on Q1) and Variant 2 (wrong Q1)
        await client.post("/api/v1/submissions/batch-sync", json={
            "assignment_id": "TEST-MV-01",
            "class_id": "cls_7a_test",
            "submissions": [
                {
                    "student_id": "stu_a1",
                    "student_name": "=SUM(A1) Амир",
                    "variant": 1,
                    "overall_score": 2.0,
                    "max_score": 2.0,
                    "final_grade": 5,
                    "questions_results": [
                        {
                            "question_number": 1,
                            "marker_id": 11,
                            "topic_tag": "case_dative",
                            "is_correct": True,
                            "points_earned": 1.0,
                            "cells": [{"cell_index": i, "expected_char": c, "predicted_char": ("О" if i == 0 else c), "confidence": 0.90, "status": "FLAG_OVERRIDDEN_BY_TEACHER" if i == 0 else "MATCH", "teacher_override": "CORRECT" if i == 0 else None} for i, c in enumerate("КИТАПКА")],
                        },
                        {
                            "question_number": 2,
                            "marker_id": 12,
                            "topic_tag": "plural_affixes",
                            "is_correct": True,
                            "points_earned": 1.0,
                            "cells": [{"cell_index": i, "expected_char": c, "predicted_char": c, "confidence": 0.95, "status": "MATCH"} for i, c in enumerate("БАЛАЛАР")],
                        },
                    ],
                },
                {
                    "student_id": "stu_a2",
                    "student_name": "Закирова Дилә Нияз кызы",
                    "variant": 2,
                    "overall_score": 1.0,
                    "max_score": 2.0,
                    "final_grade": 2,
                    "questions_results": [
                        {
                            "question_number": 1,
                            "marker_id": 11,
                            "topic_tag": "case_ablative",
                            "is_correct": False,
                            "points_earned": 0.0,
                            "cells": [{"cell_index": i, "expected_char": c, "predicted_char": "А", "confidence": 0.90, "status": "MISMATCH"} for i, c in enumerate("ӨСТӘЛДӘН")],
                        },
                        {
                            "question_number": 2,
                            "marker_id": 12,
                            "topic_tag": "plural_affixes",
                            "is_correct": True,
                            "points_earned": 1.0,
                            "cells": [{"cell_index": i, "expected_char": c, "predicted_char": c, "confidence": 0.95, "status": "MATCH"} for i, c in enumerate("УРМАННАР")],
                        },
                    ],
                },
            ],
        })

        # teacher_override="CORRECT" must NOT create problematic_letters for stu_a1
        stu_a1 = (await client.get("/api/v1/analytics/students/stu_a1")).json()
        assert stu_a1["problematic_letters"] == []
        assert stu_a1["frequent_weak_topics"] == []
        assert stu_a1["history"][0]["assignment_title"] == "Мультивариант тест"
        # Only 14 active cells (EMPTY_MATCH padding cell excluded)
        assert stu_a1["handwriting_metrics"]["total_characters_analyzed"] == 14

        # Multi-variant item analysis must keep Variant 1 Q1 and Variant 2 Q1 separate
        asm = (await client.get("/api/v1/analytics/assignments/TEST-MV-01")).json()
        by_vq = {(q["variant_id"], q["question_number"]): q for q in asm["questions_analytics"]}
        assert by_vq[(1, 1)]["prompt"] == "V1 Q1 китап ->"
        assert by_vq[(1, 1)]["expected_answer"] == "КИТАПКА"
        assert by_vq[(1, 1)]["accuracy_pct"] == 100.0
        assert by_vq[(2, 1)]["prompt"] == "V2 Q1 өстәл ->"
        assert by_vq[(2, 1)]["expected_answer"] == "ӨСТӘЛДӘН"
        assert by_vq[(2, 1)]["accuracy_pct"] == 0.0

    @pytest.mark.asyncio
    async def test_ac5_reports_sheet_title_sanitization_formula_escape_and_404(self, client: AsyncClient):
        assert (await client.get("/api/v1/reports/assignments/missing/gradebook.xlsx?class_id=cls_7a_test")).status_code == 404
        assert (await client.get("/api/v1/reports/assignments/TEST-MV-01/gradebook.xlsx?class_id=missing")).status_code == 404

        await client.post("/api/v1/submissions", json={
            "assignment_id": "TEST-MV-01",
            "class_id": "cls_7a_test",
            "submission": {
                "student_id": "stu_a1",
                "student_name": "=SUM(A1) Амир",
                "variant": 1,
                "overall_score": 2.0,
                "max_score": 2.0,
                "final_grade": 5,
                "questions_results": [],
            },
        })

        res_xlsx = await client.get("/api/v1/reports/assignments/TEST-MV-01/gradebook.xlsx?class_id=cls_7a_test&format=xlsx")
        assert res_xlsx.status_code == 200
        wb = openpyxl.load_workbook(io.BytesIO(res_xlsx.content))
        assert wb.sheetnames[0] == "Журнал 7_А"
        ws = wb.active
        assert ws.cell(row=5, column=1).value == "'=SUM(A1) Амир"

        res_csv = await client.get("/api/v1/reports/assignments/TEST-MV-01/gradebook.xlsx?class_id=cls_7a_test&format=csv")
        assert res_csv.status_code == 200
        assert "'=SUM(A1) Амир" in res_csv.text

    @pytest.mark.asyncio
    async def test_ac6_classes_bulk_import_and_assignment_validation(self, client: AsyncClient):
        # Blank import must not create ghost class
        blank = await client.post("/api/v1/classes/ghost_cls/students/bulk-import", json={"raw_text": "   \n  "})
        assert blank.status_code == 201
        assert blank.json()["added_count"] == 0
        assert (await client.get("/api/v1/classes/ghost_cls/students")).status_code == 404

        # 4-token Tatar patronymic + deduplication
        imp = await client.post(
            "/api/v1/classes/cls_new_7b/students/bulk-import",
            json={"raw_text": "Сафина Дилә Нияз кызы\nСафина Дилә Нияз кызы"},
        )
        assert imp.status_code == 201
        assert imp.json()["added_count"] == 1
        roster = (await client.get("/api/v1/classes/cls_new_7b/students")).json()["students"]
        assert roster[0]["middle_name"] == "Нияз кызы"

        # Invalid due_date and status rejected with 422
        bad_date = await client.post("/api/v1/classes/cls_new_7b/assignments", json={
            "assignment_id": "TEST-MV-01",
            "due_date": "not-a-date",
        })
        assert bad_date.status_code == 422

        bad_status = await client.post("/api/v1/classes/cls_new_7b/assignments", json={
            "assignment_id": "TEST-MV-01",
            "status": "unknown_state",
        })
        assert bad_status.status_code == 422
