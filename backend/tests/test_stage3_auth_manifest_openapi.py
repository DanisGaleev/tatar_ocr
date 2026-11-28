import base64
import hashlib
import json
from pathlib import Path

import pytest
import pytest_asyncio
import yaml
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.database import Base, get_db
from app.generators.phonetics import build_expected_cells
from app.main import app
from app.models.school import ClassAssignmentModel, ClassModel, StudentModel
from app.models.submission import SubmissionModel
from app.models.teacher import TeacherModel
from app.models.test import AssembledTestModel

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "pipeline_spec.json"
OPENAPI_YAML_PATH = REPO_ROOT / "openapi_specification.yaml"
API_MD_PATH = REPO_ROOT / "api_specification.md"

TEACHER_A = "teacher-uuid-aaaa-1111"
TEACHER_B = "teacher-uuid-bbbb-2222"

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
async def prepare_stage3_db():
    app.dependency_overrides[get_db] = override_get_db
    prev_mock = settings.TASK_EXTRACTOR_MOCK_MODE
    prev_key = settings.YANDEX_API_KEY
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        session.add_all(
            [
                TeacherModel(
                    teacher_uuid=TEACHER_A,
                    device_os="android",
                    app_version="1.0.4",
                    teacher_name="Teacher A",
                    school_name="School A",
                    grade_5_min_pct=85,
                    grade_4_min_pct=70,
                    grade_3_min_pct=50,
                    confidence_flag_threshold=0.65,
                ),
                TeacherModel(
                    teacher_uuid=TEACHER_B,
                    device_os="android",
                    app_version="1.0.4",
                    teacher_name="Teacher B",
                    school_name="School B",
                    grade_5_min_pct=85,
                    grade_4_min_pct=70,
                    grade_3_min_pct=50,
                    confidence_flag_threshold=0.65,
                ),
            ]
        )
        session.add_all(
            [
                ClassModel(
                    class_id="cls_a",
                    teacher_uuid=TEACHER_A,
                    name="7-А",
                    subject="tatar_language",
                    academic_year="2026-2027",
                ),
                ClassModel(
                    class_id="cls_b",
                    teacher_uuid=TEACHER_B,
                    name="7-Б",
                    subject="tatar_language",
                    academic_year="2026-2027",
                ),
            ]
        )
        session.add_all(
            [
                StudentModel(
                    student_id="stu_a",
                    class_id="cls_a",
                    last_name="Алиев",
                    first_name="Камил",
                    middle_name="",
                    full_name="Алиев Камил",
                    short_name="Алиев К.",
                ),
                StudentModel(
                    student_id="stu_b",
                    class_id="cls_b",
                    last_name="Сафина",
                    first_name="Алсу",
                    middle_name="",
                    full_name="Сафина Алсу",
                    short_name="Сафина А.",
                ),
            ]
        )
        bundle = {
            "assignment_id": "ASM-ST3",
            "title": "Stage 3 Isolation Test",
            "total_variants": 1,
            "variants": [
                {
                    "variant_id": 1,
                    "qr_signature": '{"tid":"ASM-ST3","var":1,"page":1,"tot":1,"n_q":1}',
                    "questions": [
                        {
                            "question_number": 1,
                            "marker_id": 11,
                            "prompt": "Китап ->",
                            "topic_tag": "case_dative",
                            "topic_name_tt": "Юнәлеш килеше",
                            "cell_count": 8,
                            "expected_answer": "КИТАПКА",
                            "expected_cells": build_expected_cells("КИТАПКА", 8),
                        }
                    ],
                }
            ],
        }
        session.add(
            AssembledTestModel(
                test_id="ASM-ST3",
                title="Stage 3 Isolation Test",
                grade_level=7,
                total_variants=1,
                bundle_json=json.dumps(bundle, ensure_ascii=False),
            )
        )
        session.add(
            ClassAssignmentModel(
                id="ca_a_st3",
                class_id="cls_a",
                assignment_id="ASM-ST3",
                status="active",
            )
        )
        session.add(
            SubmissionModel(
                submission_id="sub_a_st3",
                client_submission_uuid="cli-sub-a-st3",
                assignment_id="ASM-ST3",
                class_id="cls_a",
                student_id="stu_a",
                student_name="Алиев Камил",
                variant=1,
                overall_score=1.0,
                max_score=1.0,
                final_grade=5,
                teacher_reviewed_flags=0,
                questions_results_json=json.dumps(
                    [
                        {
                            "question_number": 1,
                            "marker_id": 11,
                            "topic_tag": "case_dative",
                            "is_correct": True,
                            "points_earned": 1.0,
                            "cells": [
                                {
                                    "cell_index": 0,
                                    "expected_char": "К",
                                    "predicted_char": "К",
                                    "confidence": 0.95,
                                    "status": "MATCH",
                                }
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
            )
        )
        await session.commit()

    yield

    settings.TASK_EXTRACTOR_MOCK_MODE = prev_mock
    settings.YANDEX_API_KEY = prev_key
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestStage3AuthIsolationManifestOpenAPI:
    """AC-1 through AC-6 for Stage 3."""

    @pytest.mark.asyncio
    async def test_ac1_device_handshake_validation_and_threshold_persistence(self, client: AsyncClient):
        base_payload = {
            "device_os": "android",
            "app_version": "1.1.0",
            "teacher_name": "Нурмиева Лилия",
            "school_name": "Лицей №1",
            "grading_scale": {"grade_5_min_pct": 85, "grade_4_min_pct": 70, "grade_3_min_pct": 50},
        }

        blank_uuid = await client.post(
            "/api/v1/auth/device-handshake",
            headers={"X-Teacher-UUID": "   "},
            json=base_payload,
        )
        assert blank_uuid.status_code == 422

        bad_scale = dict(base_payload, grading_scale={"grade_5_min_pct": 70, "grade_4_min_pct": 70, "grade_3_min_pct": 50})
        res_bad_scale = await client.post(
            "/api/v1/auth/device-handshake",
            headers={"X-Teacher-UUID": "teacher-new-1"},
            json=bad_scale,
        )
        assert res_bad_scale.status_code == 422

        out_of_range_scale = dict(
            base_payload, grading_scale={"grade_5_min_pct": 105, "grade_4_min_pct": 70, "grade_3_min_pct": 50}
        )
        res_range = await client.post(
            "/api/v1/auth/device-handshake",
            headers={"X-Teacher-UUID": "teacher-new-1"},
            json=out_of_range_scale,
        )
        assert res_range.status_code == 422

        payload_with_thresh = dict(base_payload, confidence_flag_threshold=0.72)
        res_create = await client.post(
            "/api/v1/auth/device-handshake",
            headers={"X-Teacher-UUID": "teacher-new-1"},
            json=payload_with_thresh,
        )
        assert res_create.status_code == 200
        assert res_create.json()["preferences"]["confidence_flag_threshold"] == 0.72

        # Subsequent handshake omitting confidence_flag_threshold preserves existing value
        res_update = await client.post(
            "/api/v1/auth/device-handshake",
            headers={"X-Teacher-UUID": "teacher-new-1"},
            json=base_payload,
        )
        assert res_update.status_code == 200
        assert res_update.json()["preferences"]["confidence_flag_threshold"] == 0.72

    @pytest.mark.asyncio
    async def test_ac2_multi_teacher_isolation_across_routers(self, client: AsyncClient):
        headers_a = {"X-Teacher-UUID": TEACHER_A}
        headers_b = {"X-Teacher-UUID": TEACHER_B}

        # Whitespace header rejected cleanly across scoped endpoints
        assert (await client.get("/api/v1/classes", headers={"X-Teacher-UUID": "   "})).status_code == 422

        # Teacher B sees only cls_b in class listing
        cls_list_b = await client.get("/api/v1/classes", headers=headers_b)
        assert cls_list_b.status_code == 200
        assert [c["class_id"] for c in cls_list_b.json()["classes"]] == ["cls_b"]

        # Cross-teacher access to Teacher A resources is denied (403 Forbidden)
        assert (await client.get("/api/v1/classes/cls_a/students", headers=headers_b)).status_code == 403
        assert (
            await client.post(
                "/api/v1/classes/cls_a/students/bulk-import",
                headers=headers_b,
                json={"raw_text": "Халиуллин Булат"},
            )
        ).status_code == 403
        assert (
            await client.post(
                "/api/v1/classes/cls_a/assignments",
                headers=headers_b,
                json={"assignment_id": "ASM-ST3"},
            )
        ).status_code == 403
        assert (await client.get("/api/v1/classes/cls_a/assignments", headers=headers_b)).status_code == 403
        assert (
            await client.get("/api/v1/classes/cls_a/assignments/ASM-ST3/batch-blanks.pdf", headers=headers_b)
        ).status_code == 403
        assert (
            await client.get("/api/v1/assignments/ASM-ST3/batch-blanks.pdf?class_id=cls_a", headers=headers_b)
        ).status_code == 403
        assert (
            await client.get("/api/v1/assignments/ASM-ST3/blank.pdf?variant=1&student_id=stu_a", headers=headers_b)
        ).status_code == 403
        sub_payload = {
            "assignment_id": "ASM-ST3",
            "class_id": "cls_a",
            "submission": {
                "client_submission_uuid": "cli-sub-cross",
                "student_id": "stu_a",
                "student_name": "Алиев Камил",
                "variant": 1,
                "overall_score": 1.0,
                "max_score": 1.0,
                "final_grade": 5,
                "questions_results": [],
            },
        }
        assert (await client.post("/api/v1/submissions", headers=headers_b, json=sub_payload)).status_code == 403
        assert (
            await client.post(
                "/api/v1/submissions/batch-sync",
                headers=headers_b,
                json={"assignment_id": "ASM-ST3", "class_id": "cls_a", "submissions": [sub_payload["submission"]]},
            )
        ).status_code == 403
        assert (await client.get("/api/v1/submissions?class_id=cls_a", headers=headers_b)).status_code == 403
        assert (await client.get("/api/v1/analytics/students/stu_a", headers=headers_b)).status_code == 403
        assert (await client.get("/api/v1/analytics/classes/cls_a", headers=headers_b)).status_code == 403
        assert (
            await client.get("/api/v1/reports/assignments/ASM-ST3/gradebook.xlsx?class_id=cls_a", headers=headers_b)
        ).status_code == 403

        # List/summary endpoints filter out Teacher A rows for Teacher B
        subs_b = await client.get("/api/v1/submissions", headers=headers_b)
        assert subs_b.status_code == 200
        assert subs_b.json() == []

        asm_ana_b = await client.get("/api/v1/analytics/assignments/ASM-ST3", headers=headers_b)
        assert asm_ana_b.status_code == 200
        assert asm_ana_b.json()["total_submissions"] == 0

        asm_list_b = await client.get("/api/v1/assignments", headers=headers_b)
        assert asm_list_b.status_code == 200
        assert asm_list_b.json()[0]["assigned_classes"] == []

        # Teacher A retains full 200 access to all own resources
        assert (await client.get("/api/v1/classes/cls_a/students", headers=headers_a)).status_code == 200
        assert (await client.get("/api/v1/analytics/students/stu_a", headers=headers_a)).status_code == 200
        assert (await client.get("/api/v1/analytics/classes/cls_a", headers=headers_a)).status_code == 200
        assert (await client.get("/api/v1/analytics/assignments/ASM-ST3", headers=headers_a)).json()[
            "total_submissions"
        ] == 1
        assert (
            await client.get("/api/v1/reports/assignments/ASM-ST3/gradebook.xlsx?class_id=cls_a", headers=headers_a)
        ).status_code == 200

    @pytest.mark.asyncio
    async def test_ac3_model_manifest_ssot_from_pipeline_spec(self, client: AsyncClient):
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        onnx_path = REPO_ROOT / spec["model"]["default_onnx"]
        expected_sha256 = hashlib.sha256(onnx_path.read_bytes()).hexdigest()

        res = await client.get("/api/v1/model/manifest")
        assert res.status_code == 200
        body = res.json()
        assert body["model_name"] == onnx_path.stem
        assert body["model_version"] == str(spec["version"])
        assert body["num_classes"] == 39
        assert body["alphabet_classes"] == list(spec["model"]["alphabet"])
        assert body["sha256"] == expected_sha256
        assert body["preprocessing"]["input_resolution"] == [
            spec["cell"]["canonical_canvas_size"],
            spec["cell"]["canonical_canvas_size"],
        ]
        assert body["preprocessing"]["margin_trim_pct"] == float(spec["cell"]["default_margin_trim_pct"])
        assert body["preprocessing"]["target_occupancy_px"] == float(spec["cell"]["canonical_target_dim"])

    @pytest.mark.asyncio
    async def test_ac4_stage2_review_minor_tails_fixed(self, client: AsyncClient):
        assert (await client.get("/api/v1/assignments/ASM-ST3/offline-bundle?variant=0")).status_code == 422
        assert (await client.get("/api/v1/constructor/tasks?max_cells=0")).status_code == 422

        settings.TASK_EXTRACTOR_MOCK_MODE = False
        settings.YANDEX_API_KEY = ""
        res_scan = await client.post(
            "/api/v1/constructor/scan-task-base64",
            json={
                "image_base64": base64.b64encode(b"fake-png").decode("ascii"),
                "mime_type": "image/png",
            },
        )
        assert res_scan.status_code == 503
        assert "YANDEX_API_KEY" in res_scan.json()["detail"]

    def test_ac5_openapi_yaml_and_api_spec_md_parity(self):
        live = app.openapi()
        yaml_spec = yaml.safe_load(OPENAPI_YAML_PATH.read_text(encoding="utf-8"))

        live_ops = {
            (path, method.lower())
            for path, methods in live["paths"].items()
            for method in methods
            if method.lower() in {"get", "post", "put", "patch", "delete"}
        }
        yaml_ops = {
            (path, method.lower())
            for path, methods in yaml_spec["paths"].items()
            for method in methods
            if method.lower() in {"get", "post", "put", "patch", "delete"}
        }
        assert live_ops == yaml_ops
        assert ("/api/v1/constructor/tests/{test_id}/fork", "post") not in yaml_ops
        assert "TeacherUUID" in yaml_spec["components"]["securitySchemes"]

        api_md = API_MD_PATH.read_text(encoding="utf-8")
        assert "/tests/{test_id}/fork" not in api_md
        assert "/api/v1/model/manifest" in api_md
        assert "/api/v1/constructor/scan-task-base64" in api_md
        assert "tatar_ocr_v4_handwritten39_int8" in api_md
