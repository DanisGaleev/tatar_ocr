import base64
import json
from pathlib import Path
from typing import Optional

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.database import Base, get_db
from app.main import app
from app.models.school import ClassModel, StudentModel
from app.models.submission import SubmissionModel
from app.models.teacher import TeacherModel
from app.models.test import AssembledTestModel
from app.services.blank_recognition_service import (
    BlankRecognitionService,
    get_blank_recognition_service,
)
from app.services.blank_scanner_bridge import (
    InvalidBlankImageError,
    NativeBlankScannerBridge,
    NativeScannerUnavailableError,
    RawBlankScanResult,
    RawBlankStats,
    RawQRPayload,
    RawQuestionResult,
    RawScannedCell,
    RawTopPrediction,
    _parse_qr_string,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_PHONE_BLANK = REPO_ROOT / "samples" / "test_blank_phone_photo.jpg"
BENCHMARK_BLANK_01 = REPO_ROOT / "benchmark" / "images" / "blank_01.jpg"

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


def _make_word_cells(word: str, total_cells: int = 8, low_conf_idx: Optional[int] = None) -> list[RawScannedCell]:
    cells: list[RawScannedCell] = []
    for idx in range(1, total_cells + 1):
        if idx <= len(word):
            ch = word[idx - 1]
            conf = 61.5 if low_conf_idx == idx else 99.2
            cells.append(
                RawScannedCell(
                    cell_idx=idx,
                    char=ch,
                    confidence=conf,
                    is_empty=False,
                    top3=[
                        RawTopPrediction(char=ch, confidence=conf),
                        RawTopPrediction(char="А", confidence=1.2),
                        RawTopPrediction(char="О", confidence=0.5),
                    ],
                    bbox=[220 + (idx - 1) * 100, 495, 100, 100],
                )
            )
        else:
            cells.append(
                RawScannedCell(
                    cell_idx=idx,
                    char=" ",
                    confidence=100.0,
                    is_empty=True,
                    top3=[RawTopPrediction(char=" ", confidence=100.0)],
                    bbox=[220 + (idx - 1) * 100, 495, 100, 100],
                )
            )
    return cells


class FakeBlankScannerBridge:
    def __init__(self, result: Optional[RawBlankScanResult] = None, error: Optional[Exception] = None):
        self.result = result
        self.error = error
        self.seen_sizes: list[int] = []

    def scan_image_bytes(self, image_bytes: bytes) -> RawBlankScanResult:
        self.seen_sizes.append(len(image_bytes))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _sample_raw_scan(
    qr_tid: Optional[str] = "TAT-2026-Q1",
    qr_var: Optional[int] = 1,
    qr_stu_id: Optional[str] = None,
    q2_word: str = "ӨСТӘЛДӘН",
    low_conf_q1_cell: Optional[int] = None,
) -> RawBlankScanResult:
    q1_cells = _make_word_cells("КИТАПКА", 8, low_conf_idx=low_conf_q1_cell)
    q2_cells = _make_word_cells(q2_word, 8)
    name_cells = _make_word_cells("ГАЛИЕВ", 15)
    qr = (
        RawQRPayload(
            raw=json.dumps({"tid": qr_tid, "var": qr_var, "page": 1, "tot": 1, "n_q": 2, "stu_id": qr_stu_id}),
            assignment_id=qr_tid,
            variant=qr_var,
            page=1,
            total_pages=1,
            questions_count=2,
            student_id=qr_stu_id,
        )
        if qr_tid is not None
        else None
    )
    return RawBlankScanResult(
        status="success",
        rectification_method="ArUco 4-Point Geometry",
        student_name="ГАЛИЕВ",
        student_name_cells=name_cells,
        questions=[
            RawQuestionResult(q_num=1, text="КИТАПКА", cells=q1_cells),
            RawQuestionResult(q_num=2, text=q2_word, cells=q2_cells),
        ],
        stats=RawBlankStats(total_letters=7 + len(q2_word), avg_confidence=97.5, questions_count=2),
        qr_payload=qr,
    )


@pytest_asyncio.fixture(autouse=True)
async def prepare_database():
    app.dependency_overrides[get_db] = override_get_db
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        teacher = TeacherModel(
            teacher_uuid="11111111-2222-3333-4444-555555555555",
            device_os="android",
            app_version="1.0.5",
            teacher_name="Каримова Г.И.",
            school_name="Гимназия №2",
            grade_5_min_pct=85,
            grade_4_min_pct=70,
            grade_3_min_pct=50,
            confidence_flag_threshold=0.75,
        )
        cls = ClassModel(
            class_id="cls_7a_2026",
            teacher_uuid="11111111-2222-3333-4444-555555555555",
            name="7-А",
            subject="tatar_language",
            academic_year="2026-2027",
        )
        stu = StudentModel(
            student_id="stu_01",
            class_id="cls_7a_2026",
            last_name="Галиев",
            first_name="Амир",
            middle_name="Рустемович",
            full_name="Галиев Амир Рустемович",
            short_name="Галиев А.",
        )
        asm_2q = AssembledTestModel(
            test_id="TAT-2026-Q1",
            title="Татар теле. 7 сыйныф. Исем килешләре",
            grade_level=7,
            total_variants=2,
            bundle_json=json.dumps(
                {
                    "assignment_id": "TAT-2026-Q1",
                    "title": "Татар теле. 7 сыйныф",
                    "total_variants": 2,
                    "variants": [
                        {
                            "variant_id": 1,
                            "qr_signature": '{"tid":"TAT-2026-Q1","var":1,"page":1,"tot":1,"n_q":2}',
                            "questions": [
                                {
                                    "question_number": 1,
                                    "marker_id": 11,
                                    "prompt": "китап -> (юнәлеш)",
                                    "topic_tag": "case_dative",
                                    "cell_count": 8,
                                    "expected_answer": "КИТАПКА",
                                },
                                {
                                    "question_number": 2,
                                    "marker_id": 12,
                                    "prompt": "өстәл -> (чыгыш)",
                                    "topic_tag": "case_ablative",
                                    "cell_count": 8,
                                    "expected_answer": "ӨСТӘЛДӘН",
                                },
                            ],
                        },
                        {
                            "variant_id": 2,
                            "qr_signature": '{"tid":"TAT-2026-Q1","var":2,"page":1,"tot":1,"n_q":2}',
                            "questions": [
                                {
                                    "question_number": 1,
                                    "marker_id": 11,
                                    "prompt": "өстәл -> (чыгыш)",
                                    "topic_tag": "case_ablative",
                                    "cell_count": 8,
                                    "expected_answer": "ӨСТӘЛДӘН",
                                },
                                {
                                    "question_number": 2,
                                    "marker_id": 12,
                                    "prompt": "китап -> (юнәлеш)",
                                    "topic_tag": "case_dative",
                                    "cell_count": 8,
                                    "expected_answer": "КИТАПКА",
                                },
                            ],
                        },
                    ],
                },
                ensure_ascii=False,
            ),
        )
        asm_8q = AssembledTestModel(
            test_id="TAT-2026-8Q",
            title="Реальный бланк (8 сорау)",
            grade_level=7,
            total_variants=1,
            bundle_json=json.dumps(
                {
                    "assignment_id": "TAT-2026-8Q",
                    "title": "Реальный бланк (8 сорау)",
                    "total_variants": 1,
                    "variants": [
                        {
                            "variant_id": 1,
                            "questions": [
                                {"question_number": 1, "marker_id": 11, "prompt": "Q1", "topic_tag": "case_dative", "cell_count": 8, "expected_answer": "КИТАПКА"},
                                {"question_number": 2, "marker_id": 12, "prompt": "Q2", "topic_tag": "case_ablative", "cell_count": 8, "expected_answer": "ӨСТӘЛДӘН"},
                                {"question_number": 3, "marker_id": 13, "prompt": "Q3", "topic_tag": "plural_affixes", "cell_count": 8, "expected_answer": "БАЛАЛАР"},
                                {"question_number": 4, "marker_id": 14, "prompt": "Q4", "topic_tag": "lexicon", "cell_count": 8, "expected_answer": "ХӘЕРЛЕ"},
                                {"question_number": 5, "marker_id": 15, "prompt": "Q5", "topic_tag": "case_genitive", "cell_count": 9, "expected_answer": "УРМАННЫҢ"},
                                {"question_number": 6, "marker_id": 16, "prompt": "Q6", "topic_tag": "lexicon", "cell_count": 8, "expected_answer": "ДӨРЕС"},
                                {"question_number": 7, "marker_id": 17, "prompt": "Q7", "topic_tag": "translation", "cell_count": 8, "expected_answer": "РӘХМӘТ"},
                                {"question_number": 8, "marker_id": 18, "prompt": "Q8", "topic_tag": "case_accusative", "cell_count": 8, "expected_answer": "ДӘФТӘРНЕ"},
                            ],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        )
        session.add_all([teacher, cls, stu, asm_2q, asm_8q])
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


class TestBlankScannerBridgeUnit:
    def test_parse_qr_string_valid_and_invalid(self):
        parsed = _parse_qr_string('{"tid":"TAT-2026-Q1","var":2,"page":1,"tot":1,"n_q":8,"stu_id":"stu_01"}')
        assert parsed is not None
        assert parsed.assignment_id == "TAT-2026-Q1"
        assert parsed.variant == 2
        assert parsed.questions_count == 8
        assert parsed.student_id == "stu_01"

        plain = _parse_qr_string("not-json-qr")
        assert plain is not None
        assert plain.raw == "not-json-qr"
        assert plain.assignment_id is None

        assert _parse_qr_string("   ") is None

    def test_native_bridge_rejects_empty_and_corrupt_bytes(self):
        bridge = NativeBlankScannerBridge()
        with pytest.raises(InvalidBlankImageError, match="empty"):
            bridge.scan_image_bytes(b"")
        with pytest.raises(InvalidBlankImageError, match="decode"):
            bridge.scan_image_bytes(b"not_an_image_payload")

    def test_native_bridge_missing_weights_raises_unavailable(self):
        bridge = NativeBlankScannerBridge(model_path="models/nonexistent_model.onnx")
        with pytest.raises(NativeScannerUnavailableError, match="ONNX model weights not found"):
            bridge._get_scanner()


class TestBlankRecognitionServiceAndAPI:
    @pytest.mark.asyncio
    async def test_recognize_multipart_qr_auto_resolve_and_full_parity(self, client: AsyncClient):
        fake_bridge = FakeBlankScannerBridge(_sample_raw_scan(qr_tid="TAT-2026-Q1", qr_var=1, qr_stu_id="stu_01", low_conf_q1_cell=5))
        app.dependency_overrides[get_blank_recognition_service] = lambda: BlankRecognitionService(bridge=fake_bridge)

        files = {"file": ("blank.jpg", b"fake-jpeg-bytes", "image/jpeg")}
        res = await client.post(
            "/api/v1/submissions/recognize",
            files=files,
            headers={"X-Teacher-UUID": "11111111-2222-3333-4444-555555555555"},
        )
        assert res.status_code == 200
        data = res.json()

        # Local scanner fields
        assert data["status"] == "success"
        assert data["rectification_method"] == "ArUco 4-Point Geometry"
        assert data["student_name"] == "ГАЛИЕВ"
        assert len(data["student_name_cells"]) == 15
        assert len(data["questions"]) == 2
        assert data["questions"][0]["text"] == "КИТАПКА"
        assert data["questions"][0]["cells"][0]["top3"][0] == {"char": "К", "confidence": 99.2}
        assert data["questions"][0]["cells"][0]["bbox"] == [220, 495, 100, 100]

        # Teacher threshold (0.75 -> 75.0%) flagged cell 5 (61.5%)
        assert data["confidence_threshold_pct"] == 75.0
        assert data["questions"][0]["cells"][4]["review_required"] is True
        assert data["stats"]["uncertain_cells_count"] == 1
        assert data["stats"]["zero_touch"] is False
        assert data["teacher_reviewed_flags"] == 1

        # QR + DB grading fields
        assert data["qr_metadata"]["assignment_id"] == "TAT-2026-Q1"
        assert data["assignment_id"] == "TAT-2026-Q1"
        assert data["variant"] == 1
        assert data["student_id"] == "stu_01"
        assert data["class_id"] == "cls_7a_2026"
        assert data["is_verified"] is True
        assert data["overall_score"] == 2.0
        assert data["max_score"] == 2.0
        assert data["score_percentage"] == 100.0
        assert data["final_grade"] == 5

        # Submission draft compatibility with POST /api/v1/submissions
        draft = data["submission_draft"]
        assert draft is not None
        assert draft["questions_results"][0]["cells"][0]["cell_index"] == 0  # 0-based
        assert 0.0 <= draft["questions_results"][0]["cells"][0]["confidence"] <= 1.0
        save_res = await client.post(
            "/api/v1/submissions",
            json={"assignment_id": data["assignment_id"], "class_id": data["class_id"], "submission": draft},
        )
        assert save_res.status_code == 200
        assert save_res.json()["status"] == "saved"

    @pytest.mark.asyncio
    async def test_recognize_base64_with_auto_save_and_explicit_override(self, client: AsyncClient):
        # Q2 has wrong answer ӨСТӘЛДАН -> 1/2 = 50% -> Grade 3
        fake_bridge = FakeBlankScannerBridge(_sample_raw_scan(qr_tid="TAT-2026-Q1", qr_var=2, q2_word="ӨСТӘЛДАН"))
        app.dependency_overrides[get_blank_recognition_service] = lambda: BlankRecognitionService(bridge=fake_bridge)

        payload = {
            "image_base64": base64.b64encode(b"fake-png-bytes").decode("ascii"),
            "mime_type": "image/png",
            "assignment_id": "TAT-2026-Q1",
            "variant": 1,  # overrides QR var=2
            "student_id": "stu_01",
            "auto_save": True,
        }
        res = await client.post("/api/v1/submissions/recognize-base64", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["variant"] == 1
        assert data["is_verified"] is True
        assert data["overall_score"] == 1.0
        assert data["max_score"] == 2.0
        assert data["final_grade"] == 3
        assert data["stats"]["zero_touch"] is True
        assert data["saved_submission_id"] is not None

        async with TestSessionLocal() as session:
            saved = (
                await session.execute(
                    select(SubmissionModel).where(SubmissionModel.submission_id == data["saved_submission_id"])
                )
            ).scalar_one_or_none()
            assert saved is not None
            assert saved.overall_score == 1.0
            assert saved.final_grade == 3

    @pytest.mark.asyncio
    async def test_recognize_validation_and_error_cases(self, client: AsyncClient):
        fake_bridge = FakeBlankScannerBridge(_sample_raw_scan(qr_tid="UNKNOWN-QR-ID"))
        app.dependency_overrides[get_blank_recognition_service] = lambda: BlankRecognitionService(bridge=fake_bridge)

        # 1. Invalid content-type -> 400
        res = await client.post(
            "/api/v1/submissions/recognize",
            files={"file": ("doc.txt", b"hello", "text/plain")},
        )
        assert res.status_code == 400

        # 2. Empty file -> 400
        res = await client.post(
            "/api/v1/submissions/recognize",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert res.status_code == 400

        # 3. Oversized file -> 413
        orig_limit = settings.BLANK_MAX_IMAGE_SIZE_BYTES
        try:
            settings.BLANK_MAX_IMAGE_SIZE_BYTES = 10
            res = await client.post(
                "/api/v1/submissions/recognize",
                files={"file": ("big.jpg", b"0123456789abcdef", "image/jpeg")},
            )
            assert res.status_code == 413
        finally:
            settings.BLANK_MAX_IMAGE_SIZE_BYTES = orig_limit

        # 4. Invalid base64 -> 400
        res = await client.post(
            "/api/v1/submissions/recognize-base64",
            json={"image_base64": "%%%invalid%%%", "mime_type": "image/jpeg"},
        )
        assert res.status_code == 400

        # 5. Explicit missing assignment -> 404
        res = await client.post(
            "/api/v1/submissions/recognize",
            params={"assignment_id": "MISSING-999"},
            files={"file": ("blank.jpg", b"abc", "image/jpeg")},
        )
        assert res.status_code == 404

        # 6. Explicit missing variant -> 404
        res = await client.post(
            "/api/v1/submissions/recognize",
            params={"assignment_id": "TAT-2026-Q1", "variant": 9},
            files={"file": ("blank.jpg", b"abc", "image/jpeg")},
        )
        assert res.status_code == 404

        # 7. Unregistered QR assignment -> 200 with OCR intact and verification_warning
        res = await client.post(
            "/api/v1/submissions/recognize",
            files={"file": ("blank.jpg", b"abc", "image/jpeg")},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["is_verified"] is False
        assert "not found" in body["verification_warning"]
        assert len(body["questions"]) == 2

        # 8. auto_save=True without resolvable class/assignment -> 422
        res = await client.post(
            "/api/v1/submissions/recognize",
            params={"assignment_id": "TAT-2026-Q1", "auto_save": "true"},
            files={"file": ("blank.jpg", b"abc", "image/jpeg")},
        )
        assert res.status_code == 422

        # 9. Native scanner unavailable -> 503
        broken_bridge = FakeBlankScannerBridge(error=NativeScannerUnavailableError("missing dll"))
        app.dependency_overrides[get_blank_recognition_service] = lambda: BlankRecognitionService(bridge=broken_bridge)
        res = await client.post(
            "/api/v1/submissions/recognize",
            files={"file": ("blank.jpg", b"abc", "image/jpeg")},
        )
        assert res.status_code == 503


class TestRealBlankPhotoRecognitionE2E:
    """End-to-end integration tests running real blank photos through FastAPI + C++20 tatar_ocr_native."""

    @pytest.mark.asyncio
    async def test_real_phone_photo_multipart_with_verification_and_save(self, client: AsyncClient):
        assert REAL_PHONE_BLANK.exists(), f"Missing real sample blank: {REAL_PHONE_BLANK}"
        image_bytes = REAL_PHONE_BLANK.read_bytes()

        res = await client.post(
            "/api/v1/submissions/recognize",
            params={"assignment_id": "TAT-2026-8Q", "student_id": "stu_01", "auto_save": "true"},
            files={"file": (REAL_PHONE_BLANK.name, image_bytes, "image/jpeg")},
        )
        assert res.status_code == 200, res.text
        data = res.json()

        assert data["status"] == "success"
        assert data["rectification_method"] == "ArUco 4-Point Geometry"
        assert data["qr_metadata"] is not None
        assert data["qr_metadata"]["assignment_id"] == "TAT-2026-Q1"
        assert data["qr_metadata"]["variant"] == 1
        assert data["qr_metadata"]["questions_count"] == 8

        assert data["assignment_id"] == "TAT-2026-8Q"
        assert data["student_id"] == "stu_01"
        assert data["class_id"] == "cls_7a_2026"
        assert data["is_verified"] is True
        assert data["overall_score"] == 8.0
        assert data["max_score"] == 8.0
        assert data["final_grade"] == 5
        assert data["stats"]["questions_count"] == 8
        assert data["stats"]["total_letters"] == 55
        assert data["stats"]["zero_touch"] is True
        assert data["saved_submission_id"] is not None
        assert [q["text"] for q in data["questions"]] == [
            "КИТАПКА",
            "ӨСТӘЛДӘН",
            "БАЛАЛАР",
            "ХӘЕРЛЕ",
            "УРМАННЫҢ",
            "ДӨРЕС",
            "РӘХМӘТ",
            "ДӘФТӘРНЕ",
        ]

    @pytest.mark.asyncio
    async def test_real_benchmark_blank_01_base64_without_qr(self, client: AsyncClient):
        assert BENCHMARK_BLANK_01.exists(), f"Missing benchmark blank: {BENCHMARK_BLANK_01}"
        b64_img = base64.b64encode(BENCHMARK_BLANK_01.read_bytes()).decode("ascii")

        res = await client.post(
            "/api/v1/submissions/recognize-base64",
            json={"image_base64": b64_img, "mime_type": "image/jpeg"},
        )
        assert res.status_code == 200, res.text
        data = res.json()

        assert data["status"] == "success"
        assert data["rectification_method"] == "ArUco 4-Point Geometry"
        assert data["qr_metadata"] is None
        assert data["is_verified"] is False
        assert data["stats"]["questions_count"] == 4
        assert data["stats"]["total_letters"] == 32
        assert len(data["questions"]) == 4
        assert len(data["questions"][0]["cells"]) == 8
        assert len(data["questions"][0]["cells"][0]["top3"]) == 3
        assert len(data["questions"][0]["cells"][0]["bbox"]) == 4
