import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
import io
import json
import openpyxl

from app.main import app
from app.core.database import Base, get_db
from app.models.school import ClassModel, StudentModel
from app.models.teacher import TeacherModel
from app.models.submission import SubmissionModel
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
        yield session

@pytest_asyncio.fixture(autouse=True)
async def prepare_database():
    app.dependency_overrides[get_db] = override_get_db
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Pre-seed class, students, and an assignment
    async with TestSessionLocal() as session:
        cls = ClassModel(
            class_id="cls_7a_2026",
            teacher_uuid="c56a4180-65aa-42ec-a945-5fd21dec0538",
            name="7-А",
            subject="tatar_language",
            academic_year="2026-2027",
        )
        session.add(cls)

        s1 = StudentModel(
            student_id="stu_01",
            class_id="cls_7a_2026",
            last_name="Галиев",
            first_name="Амир",
            middle_name="Рустемович",
            full_name="Галиев Амир Рустемович",
            short_name="Галиев А.",
        )
        s2 = StudentModel(
            student_id="stu_02",
            class_id="cls_7a_2026",
            last_name="Закирова",
            first_name="Ләйсән",
            middle_name="Ильдаровна",
            full_name="Закирова Ләйсән Ильдаровна",
            short_name="Закирова Л.",
        )
        session.add_all([s1, s2])

        asm = AssembledTestModel(
            test_id="TAT-2026-Q1",
            title="Татар теле. 7 сыйныф. Исем килешләре",
            grade_level=7,
            total_variants=2,
            bundle_json=json.dumps({
                "assignment_id": "TAT-2026-Q1",
                "title": "Татар теле. 7 сыйныф. Исем килешләре",
                "total_variants": 2,
                "variants": [
                    {
                        "variant_id": 1,
                        "qr_signature": '{"tid":"TAT-2026-Q1","var":1,"page":1,"tot":1,"n_q":4}',
                        "template_geometry": {
                            "format": "A4",
                            "corner_aruco_dict": "DICT_4X4_50",
                            "corner_aruco_ids": [0, 1, 2, 3],
                            "cell_dimensions_mm": {"width": 10.0, "height": 10.0}
                        },
                        "questions": [
                            {"question_number": 1, "marker_id": 11, "prompt": "Куегыз сүзне юнәлеш килешендә: китап ->", "topic_tag": "case_dative", "topic_name_tt": "Юнәлеш килеше", "cell_count": 8, "expected_answer": "КИТАПКА", "expected_cells": [{"index": 0, "char": "К", "unicode": "U+041A", "is_empty_allowed": False}]},
                            {"question_number": 2, "marker_id": 12, "prompt": "Куегыз сүзне чыгыш килешендә: өстәл ->", "topic_tag": "case_ablative", "topic_name_tt": "Чыгыш килеше", "cell_count": 8, "expected_answer": "ӨСТӘЛДӘН", "expected_cells": [{"index": 0, "char": "Ө", "unicode": "U+04E8", "is_empty_allowed": False}]},
                            {"question_number": 3, "marker_id": 13, "prompt": "Сүзгә күплек сан кушымчасын ялгагыз: бала ->", "topic_tag": "plural_affixes", "topic_name_tt": "Күплек сан", "cell_count": 8, "expected_answer": "БАЛАЛАР", "expected_cells": [{"index": 0, "char": "Б", "unicode": "U+0411", "is_empty_allowed": False}]},
                            {"question_number": 4, "marker_id": 14, "prompt": "Сүзгә күплек сан кушымчасын ялгагыз: урман ->", "topic_tag": "plural_affixes", "topic_name_tt": "Күплек сан", "cell_count": 8, "expected_answer": "УРМАННАР", "expected_cells": [{"index": 0, "char": "У", "unicode": "U+0423", "is_empty_allowed": False}]}
                        ]
                    },
                    {
                        "variant_id": 2,
                        "qr_signature": '{"tid":"TAT-2026-Q1","var":2,"page":1,"tot":1,"n_q":4}',
                        "template_geometry": {
                            "format": "A4",
                            "corner_aruco_dict": "DICT_4X4_50",
                            "corner_aruco_ids": [0, 1, 2, 3],
                            "cell_dimensions_mm": {"width": 10.0, "height": 10.0}
                        },
                        "questions": [
                            {"question_number": 1, "marker_id": 11, "prompt": "Куегыз сүзне чыгыш килешендә: өстәл ->", "topic_tag": "case_ablative", "topic_name_tt": "Чыгыш килеше", "cell_count": 8, "expected_answer": "ӨСТӘЛДӘН", "expected_cells": [{"index": 0, "char": "Ө", "unicode": "U+04E8", "is_empty_allowed": False}]},
                            {"question_number": 2, "marker_id": 12, "prompt": "Сүзгә күплек сан кушымчасын ялгагыз: бала ->", "topic_tag": "plural_affixes", "topic_name_tt": "Күплек сан", "cell_count": 8, "expected_answer": "БАЛАЛАР", "expected_cells": [{"index": 0, "char": "Б", "unicode": "U+0411", "is_empty_allowed": False}]},
                            {"question_number": 3, "marker_id": 13, "prompt": "Сүзгә күплек сан кушымчасын ялгагыз: урман ->", "topic_tag": "plural_affixes", "topic_name_tt": "Күплек сан", "cell_count": 8, "expected_answer": "УРМАННАР", "expected_cells": [{"index": 0, "char": "У", "unicode": "U+0423", "is_empty_allowed": False}]},
                            {"question_number": 4, "marker_id": 14, "prompt": "Куегыз сүзне юнәлеш килешендә: китап ->", "topic_tag": "case_dative", "topic_name_tt": "Юнәлеш килеше", "cell_count": 8, "expected_answer": "КИТАПКА", "expected_cells": [{"index": 0, "char": "К", "unicode": "U+041A", "is_empty_allowed": False}]}
                        ]
                    }
                ]
            }, ensure_ascii=False),
        )
        session.add(asm)
        await session.commit()

    yield

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    app.dependency_overrides.clear()

@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestTeacherIdentityAndAuthAPI:
    """Tests for Router 0: /api/v1/auth/device-handshake"""

    @pytest.mark.asyncio
    async def test_handshake_creates_teacher(self, client: AsyncClient):
        teacher_uuid = "c56a4180-65aa-42ec-a945-5fd21dec0538"
        payload = {
            "device_os": "android",
            "app_version": "1.0.4",
            "teacher_name": "Каримова Гөлнара Илдар кызы",
            "school_name": "Гимназия №2 им. Ш. Марджани",
            "grading_scale": {
                "grade_5_min_pct": 85,
                "grade_4_min_pct": 70,
                "grade_3_min_pct": 50,
            }
        }
        res = await client.post(
            "/api/v1/auth/device-handshake",
            json=payload,
            headers={"X-Teacher-UUID": teacher_uuid}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "active"
        assert data["teacher_uuid"] == teacher_uuid
        assert data["teacher_name"] == "Каримова Гөлнара Илдар кызы"
        assert data["preferences"]["grading_scale"]["grade_5_min_pct"] == 85
        assert data["preferences"]["confidence_flag_threshold"] == 0.65

    @pytest.mark.asyncio
    async def test_handshake_updates_existing_teacher(self, client: AsyncClient):
        teacher_uuid = "c56a4180-65aa-42ec-a945-5fd21dec0538"
        # First call
        await client.post(
            "/api/v1/auth/device-handshake",
            json={
                "device_os": "android",
                "app_version": "1.0.0",
                "teacher_name": "Каримова Г.",
                "school_name": "Мәктәп",
                "grading_scale": {"grade_5_min_pct": 85, "grade_4_min_pct": 70, "grade_3_min_pct": 50}
            },
            headers={"X-Teacher-UUID": teacher_uuid}
        )
        # Update call
        res = await client.post(
            "/api/v1/auth/device-handshake",
            json={
                "device_os": "android",
                "app_version": "1.0.5",
                "teacher_name": "Каримова Гөлнара Илдар кызы",
                "school_name": "Гимназия №2",
                "grading_scale": {"grade_5_min_pct": 90, "grade_4_min_pct": 75, "grade_3_min_pct": 55}
            },
            headers={"X-Teacher-UUID": teacher_uuid}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["preferences"]["grading_scale"]["grade_5_min_pct"] == 90


class TestClassesAndStudentsAPI:
    """Tests for Router 1: /api/v1/classes"""

    @pytest.mark.asyncio
    async def test_list_classes(self, client: AsyncClient):
        res = await client.get("/api/v1/classes", headers={"X-Teacher-UUID": "c56a4180-65aa-42ec-a945-5fd21dec0538"})
        assert res.status_code == 200
        data = res.json()
        assert "classes" in data
        assert len(data["classes"]) >= 1
        cls = data["classes"][0]
        assert cls["class_id"] == "cls_7a_2026"
        assert cls["student_count"] == 2

    @pytest.mark.asyncio
    async def test_get_class_roster(self, client: AsyncClient):
        res = await client.get("/api/v1/classes/cls_7a_2026/students")
        assert res.status_code == 200
        data = res.json()
        assert data["class_id"] == "cls_7a_2026"
        assert len(data["students"]) == 2
        names = [s["short_name"] for s in data["students"]]
        assert "Галиев А." in names
        assert "Закирова Л." in names

    @pytest.mark.asyncio
    async def test_bulk_import_students(self, client: AsyncClient):
        raw_text = "Хабибуллин Тимур Маратович\nСафина Дилә Нияз кызы"
        res = await client.post(
            "/api/v1/classes/cls_7a_2026/students/bulk-import",
            json={"raw_text": raw_text},
            headers={"X-Teacher-UUID": "c56a4180-65aa-42ec-a945-5fd21dec0538"}
        )
        assert res.status_code == 201
        data = res.json()
        assert data["added_count"] == 2
        assert len(data["students"]) == 2

        # Verify roster now has 4 students
        res2 = await client.get("/api/v1/classes/cls_7a_2026/students")
        assert len(res2.json()["students"]) == 4


class TestSubmissionsAndAnalyticsAPI:
    """Tests for Router 4 (Submissions) & Router 5 (Analytics) & Router 6 (Reports)"""

    @pytest.mark.asyncio
    async def test_zero_photo_batch_sync_and_analytics(self, client: AsyncClient):
        batch_payload = {
            "assignment_id": "TAT-2026-Q1",
            "class_id": "cls_7a_2026",
            "synced_at": "2026-09-18T13:00:00Z",
            "submissions": [
                {
                    "client_submission_uuid": "sync-sub-01",
                    "student_id": "stu_01",
                    "student_name": "Галиев Амир Рустемович",
                    "variant": 1,
                    "overall_score": 8.0,
                    "max_score": 8.0,
                    "final_grade": 5,
                    "teacher_reviewed_flags": 0,
                    "questions_results": [
                        {
                            "question_number": 1,
                            "marker_id": 11,
                            "topic_tag": "case_dative",
                            "is_correct": True,
                            "points_earned": 1.0,
                            "cells": [
                                {"cell_index": 0, "expected_char": "К", "predicted_char": "К", "confidence": 0.98, "status": "MATCH"},
                                {"cell_index": 1, "expected_char": "И", "predicted_char": "И", "confidence": 0.95, "status": "MATCH"},
                            ]
                        }
                    ]
                },
                {
                    "client_submission_uuid": "sync-sub-02",
                    "student_id": "stu_02",
                    "student_name": "Закирова Ләйсән Ильдаровна",
                    "variant": 1,
                    "overall_score": 4.0,
                    "max_score": 8.0,
                    "final_grade": 3,
                    "teacher_reviewed_flags": 1,
                    "questions_results": [
                        {
                            "question_number": 1,
                            "marker_id": 11,
                            "topic_tag": "case_ablative",
                            "is_correct": False,
                            "points_earned": 0.0,
                            "cells": [
                                {"cell_index": 0, "expected_char": "Ө", "predicted_char": "Ө", "confidence": 0.92, "status": "MATCH"},
                                # OCR confusion: expected Ң but predicted Н
                                {"cell_index": 1, "expected_char": "Ң", "predicted_char": "Н", "confidence": 0.52, "status": "FLAG_OVERRIDDEN_BY_TEACHER", "teacher_override": "WRONG"},
                            ]
                        }
                    ]
                }
            ]
        }

        # 1. Batch sync
        res_sync = await client.post("/api/v1/submissions/batch-sync", json=batch_payload)
        assert res_sync.status_code == 200
        sync_data = res_sync.json()
        assert sync_data["status"] == "synced"
        assert sync_data["received_count"] == 2
        assert sync_data["inserted_count"] == 2
        assert sync_data["analytics_recalculated"] is True

        # 2. Query submissions
        res_q = await client.get("/api/v1/submissions?class_id=cls_7a_2026")
        assert res_q.status_code == 200
        assert len(res_q.json()) == 2

        # 3. Test Student Analytics
        res_stu = await client.get("/api/v1/analytics/students/stu_02")
        assert res_stu.status_code == 200
        stu_data = res_stu.json()
        assert stu_data["student_id"] == "stu_02"
        assert stu_data["total_tests_completed"] == 1
        assert stu_data["average_score_pct"] == 50.0
        assert stu_data["average_grade"] == 3.0
        # Verify problematic letters picked up Ң vs Н confusion
        assert len(stu_data["problematic_letters"]) >= 1
        prob = stu_data["problematic_letters"][0]
        assert prob["letter"] == "Ң"
        assert "Н" in prob["common_confusions"]

        # 4. Test Class Analytics & Heatmap
        res_cls = await client.get("/api/v1/analytics/classes/cls_7a_2026")
        assert res_cls.status_code == 200
        cls_data = res_cls.json()
        assert cls_data["class_id"] == "cls_7a_2026"
        assert cls_data["average_class_score_pct"] == 75.0  # (100% + 50%) / 2
        assert cls_data["grade_distribution"]["5"] == 1
        assert cls_data["grade_distribution"]["3"] == 1
        # Top mistakes should show case_ablative with 100% failure rate
        assert len(cls_data["top_class_mistakes"]) >= 1
        assert cls_data["top_class_mistakes"][0]["topic_code"] == "case_ablative"
        assert cls_data["top_class_mistakes"][0]["failure_rate_pct"] == 100.0

        # Leaderboard check: stu_01 (100%) should be first, stu_02 (50%) second
        leaderboard = cls_data["students_performance_table"]
        assert leaderboard[0]["student_id"] == "stu_01"
        assert leaderboard[0]["average_score_pct"] == 100.0

        # 5. Test Assignment Analytics
        res_asm = await client.get("/api/v1/analytics/assignments/TAT-2026-Q1")
        assert res_asm.status_code == 200
        asm_data = res_asm.json()
        assert asm_data["total_submissions"] == 2
        assert len(asm_data["questions_analytics"]) >= 1

        # 6. Test Gradebook Reports (CSV and Excel .xlsx)
        # CSV export
        res_csv = await client.get("/api/v1/reports/assignments/TAT-2026-Q1/gradebook.xlsx?class_id=cls_7a_2026&format=csv")
        assert res_csv.status_code == 200
        assert "text/csv" in res_csv.headers["content-type"]
        csv_text = res_csv.text
        assert "Галиев Амир Рустемович" in csv_text
        assert "Закирова Ләйсән Ильдаровна" in csv_text

        # Excel (.xlsx) export
        res_xlsx = await client.get("/api/v1/reports/assignments/TAT-2026-Q1/gradebook.xlsx?class_id=cls_7a_2026&format=xlsx")
        assert res_xlsx.status_code == 200
        assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in res_xlsx.headers["content-type"]
        
        # Verify valid Excel binary using openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(res_xlsx.content))
        assert "Журнал 7-А" in wb.sheetnames
        ws = wb["Журнал 7-А"]
        # Row 1 is title, Row 4 is header, Rows 5 and 6 are students
        names_in_excel = [ws.cell(row=r, column=1).value for r in range(5, 7)]
        assert "Галиев Амир Рустемович" in names_in_excel
        assert "Закирова Ләйсән Ильдаровна" in names_in_excel

        # 7. Test Download Printable Blank PDF
        res_pdf = await client.get("/api/v1/assignments/TAT-2026-Q1/blank.pdf?variant=1")
        assert res_pdf.status_code == 200
        assert "application/pdf" in res_pdf.headers["content-type"]
        assert len(res_pdf.content) > 1000
        # Valid PDF header
        assert res_pdf.content.startswith(b"%PDF")

        # 8. Test Offline Bundle for TAT-2026-Q1 (Fixed 500 error & schema validation)
        res_bundle = await client.get("/api/v1/assignments/TAT-2026-Q1/offline-bundle")
        assert res_bundle.status_code == 200, f"offline-bundle returned {res_bundle.status_code}: {res_bundle.text}"
        bundle_json = res_bundle.json()
        assert bundle_json["assignment_id"] == "TAT-2026-Q1"
        assert bundle_json["total_variants"] == 2
        assert len(bundle_json["variants"]) == 2
        assert "template_geometry" in bundle_json["variants"][0]
        assert len(bundle_json["variants"][0]["questions"]) == 4
        assert "expected_cells" in bundle_json["variants"][0]["questions"][0]

        res_bundle_v1 = await client.get("/api/v1/assignments/TAT-2026-Q1/offline-bundle?variant=1")
        assert res_bundle_v1.status_code == 200
        bundle_v1 = res_bundle_v1.json()
        assert bundle_v1["total_variants"] == 1
        assert bundle_v1["variants"][0]["variant_id"] == 1

        # 9. Test ready-tests dynamic questions_count
        res_ready = await client.get("/api/v1/constructor/ready-tests")
        assert res_ready.status_code == 200
        ready_tests = res_ready.json()
        tat_test = next((t for t in ready_tests if t["test_id"] == "TAT-2026-Q1"), None)
        assert tat_test is not None
        assert tat_test["questions_count"] == 4

        # 10. Test Model Manifest & Authoritative Alphabet
        res_manifest = await client.get("/api/v1/model/manifest")
        assert res_manifest.status_code == 200
        manifest = res_manifest.json()
        assert manifest["num_classes"] == 39
        assert manifest["alphabet_classes"][0] == "А"
        assert manifest["alphabet_classes"][1] == "Ә"
        assert manifest["alphabet_classes"][2] == "Б"
        assert manifest["alphabet_classes"][-1] == "Я"
        assert manifest["preprocessing"]["input_resolution"] == [64, 64]
        assert manifest["preprocessing"]["margin_trim_pct"] == 11.0


