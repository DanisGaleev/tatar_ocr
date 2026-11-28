import base64
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.database import Base, get_db
from app.generators.lexicon import PLURAL_STEMS, SYNONYM_PAIRS, VERB_STEMS
from app.generators.morphology import PossessiveCaseGenerator
from app.generators.pdf_blank import render_blank_image
from app.generators.phonetics import (
    OCR_ALPHABET,
    PhoneticsError,
    inflect_comparative,
    inflect_possessive,
    is_front_vowel,
    validate_ocr_answer,
)
from app.generators.plural_affixes import PluralAffixesGenerator
from app.generators.reading_tasks import (
    TextComprehensionGenerator,
    TextTitleMainIdeaGenerator,
)
from app.generators.registry import registry
from app.generators.syntax_tasks import (
    FindErrorSentenceGenerator,
    SentenceMCClozeGenerator,
)
from app.main import app
from app.models.school import ClassModel, StudentModel
from app.models.teacher import TeacherModel
from app.services.task_extractor import TaskExtractorService, get_task_extractor_service
from app.services.task_extractor.base import (
    BaseLLMTaskClassifier,
    BaseOCRClient,
    ExtractedTaskResult,
    ExtractedTaskSubItem,
)
from app.services.task_extractor.yandex_gpt import YandexGPTClient

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "tatar_ocr_v4_handwritten39_int8.onnx"

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
async def prepare_stage2_db():
    app.dependency_overrides[get_db] = override_get_db
    prev_mock = settings.TASK_EXTRACTOR_MOCK_MODE
    settings.TASK_EXTRACTOR_MOCK_MODE = True
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        session.add(
            TeacherModel(
                teacher_uuid="teacher-st2",
                device_os="android",
                app_version="1.0.4",
                teacher_name="Test Teacher",
                school_name="School 2",
            )
        )
        session.add(
            ClassModel(
                class_id="cls_st2",
                teacher_uuid="teacher-st2",
                name="7/Б",
                subject="tatar_language",
                academic_year="2026-2027",
            )
        )
        session.add(
            StudentModel(
                student_id="stu_e2e",
                class_id="cls_st2",
                last_name="Алиев",
                first_name="Илдар",
                middle_name="",
                full_name="Алиев Илдар",
                short_name="Алиев И.",
            )
        )
        await session.commit()

    yield

    settings.TASK_EXTRACTOR_MOCK_MODE = prev_mock
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestStage2OCRAndPDFContract:
    """AC-1, AC-2, AC-7: SSOT PDF geometry -> C++ native OCR & assignments delivery."""

    def test_e2e_render_blank_image_to_native_scanner_1_to_12_cells_and_qr(self):
        from blank_pipeline import BlankOCRScanner

        cell_counts = [1, 4, 6, 8, 12]
        questions = [
            {
                "question_number": idx + 1,
                "marker_id": 11 + idx,
                "prompt": f"Бирем №{idx + 1}: текст бик озын булырга мөмкин, шуңа күрә барлык юллар да күренергә тиеш.",
                "cell_count": c_cnt,
                "expected_answer": "А" * c_cnt,
            }
            for idx, c_cnt in enumerate(cell_counts)
        ]
        img = render_blank_image(
            assignment_id="E2E-ST2",
            title="Контроль эш №2",
            variant_id=1,
            questions=questions,
            student_name="Алиев Илдар",
            student_id="stu-42",
        )

        qr_text, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
        assert qr_text
        qr_payload = json.loads(qr_text)
        assert qr_payload["tid"] == "E2E-ST2"
        assert qr_payload["stu_id"] == "stu-42"
        assert qr_payload["var"] == 1
        assert qr_payload["n_q"] == len(cell_counts)

        scanner = BlankOCRScanner(model_path=str(MODEL_PATH))
        rectified, ok, _method = scanner._native.rectify_sheet(img)
        assert ok is True
        name_cells, q_rows = scanner._native.extract_cells(rectified)
        assert len(name_cells) == 15
        assert [len(row["cells"]) for row in q_rows] == cell_counts

    @pytest.mark.asyncio
    async def test_assignments_blank_pdf_validation_and_404_guards(self, client: AsyncClient):
        asm_res = await client.post(
            "/api/v1/constructor/tests",
            json={
                "title": "Stage 2 Blank Test",
                "grade_level": 7,
                "topic_tags": ["case_inflection"],
                "questions_per_variant": 4,
                "generate_variants_count": 1,
                "seed": 11,
            },
        )
        assert asm_res.status_code == 201
        assignment_id = asm_res.json()["assignment_id"]

        bad_var = await client.get(f"/api/v1/assignments/{assignment_id}/blank.pdf?variant=0")
        assert bad_var.status_code == 422

        missing_var = await client.get(f"/api/v1/assignments/{assignment_id}/blank.pdf?variant=99")
        assert missing_var.status_code == 404

        missing_stu = await client.get(
            f"/api/v1/assignments/{assignment_id}/blank.pdf?variant=1&student_id=nonexistent-stu"
        )
        assert missing_stu.status_code == 404

        ok_pdf = await client.get(
            f"/api/v1/assignments/{assignment_id}/blank.pdf?variant=1&student_id=stu_e2e"
        )
        assert ok_pdf.status_code == 200
        assert ok_pdf.headers["content-type"].startswith("application/pdf")
        assert ok_pdf.content.startswith(b"%PDF")


class TestStage2GeneratorsAndLinguistics:
    """AC-3, AC-4, AC-7: 24-generator dictionary sweep, MC shuffling, phonetics, lexicon."""

    def test_exhaustive_24_generators_ocr_alphabet_and_bounds(self):
        supported = registry.list_supported_types()
        assert len(supported) == 24

        for entry in supported:
            gen = registry.get_generator(entry["task_type"])
            for seed in range(20):
                draft = gen.generate(seed=seed)
                assert 1 <= len(draft.expected_answer) <= draft.cell_count <= 12
                assert validate_ocr_answer(draft.expected_answer) == draft.expected_answer
                assert all(ch in OCR_ALPHABET for ch in draft.expected_answer)
                assert draft.task_type == entry["task_type"]
                assert isinstance(draft.topic_name_tt, str) and draft.topic_name_tt.strip()

    def test_find_error_cyrillic_and_mc_generators_shuffle_answers(self):
        err_gen = FindErrorSentenceGenerator()
        err_answers = {err_gen.generate(seed=s).expected_answer for s in range(20)}
        assert err_answers.issubset({"А", "Б", "В", "Г"})
        assert len(err_answers) > 1

        for gen in (
            SentenceMCClozeGenerator(),
            TextComprehensionGenerator(),
            TextTitleMainIdeaGenerator(),
        ):
            distinct = {gen.generate(seed=s).expected_answer for s in range(30)}
            assert len(distinct) > 1, f"{gen.task_type} did not vary MC option position"

    def test_possessive_case_and_registry_coverage(self):
        pc_gen = PossessiveCaseGenerator()
        for seed in range(100):
            draft = pc_gen.generate(seed=seed)
            assert len(draft.expected_answer) <= 12

        all_types = [t["task_type"] for t in registry.list_supported_types()]
        bundle = registry.assemble_variants(
            assignment_id="ST2-REG",
            title="Full Registry Coverage",
            grade_level=7,
            variants_count=3,
            seed=7,
            task_types=all_types,
            questions_per_variant=8,
            shuffle=False,
        )
        seen_tags = set()
        for variant in bundle["variants"]:
            prompts = [q["prompt"] for q in variant["questions"]]
            assert len(prompts) == len(set(prompts))
            for q in variant["questions"]:
                seen_tags.add(q["topic_tag"])
        assert len(seen_tags) >= 20

    def test_phonetics_and_lexicon_fixes(self):
        assert is_front_vowel("счёт") is True
        assert is_front_vowel("актёр") is True
        assert is_front_vowel("ёлка") is False
        assert is_front_vowel("каен") is False
        assert inflect_comparative("ак") == "АГЫРАК"
        assert inflect_comparative("карт") == "КАРТЫРАК"
        assert inflect_possessive("борын", 1) == "БОРНЫМ"
        assert inflect_possessive("ипи", 3) == "ИПИЕ"
        with pytest.raises(PhoneticsError):
            inflect_possessive("китап", 1, number="pl")

        assert "төзәт" in VERB_STEMS
        assert "төзә" not in VERB_STEMS
        assert ("тиз", "ҖИТЕЗ") in SYNONYM_PAIRS
        assert ("тиз", "ҖӘЛДӘМ") not in SYNONYM_PAIRS
        assert "куян" in PLURAL_STEMS
        assert PluralAffixesGenerator().generate(stems=["куян"], seed=0).expected_answer == "КУЯННАР"


class TestStage2ConstructorAndExtractor:
    """AC-5 & AC-6: constructor routes/schemas and task_extractor safety."""

    @pytest.mark.asyncio
    async def test_constructor_generate_validation_and_assemble_order_seed(self, client: AsyncClient):
        bad_case = await client.post(
            "/api/v1/constructor/generate",
            json={"task_type": "antonyms", "count": 1, "case_code": "case_dative"},
        )
        assert bad_case.status_code == 422

        bad_stems = await client.post(
            "/api/v1/constructor/generate",
            json={"task_type": "antonyms", "count": 1, "custom_stems": ["китап"]},
        )
        assert bad_stems.status_code == 422

        gen_res = await client.post(
            "/api/v1/constructor/generate",
            json={
                "task_type": "plural_affixes",
                "count": 3,
                "grade_level": 6,
                "seed": 42,
                "save_to_bank": True,
            },
        )
        assert gen_res.status_code == 200
        tasks = gen_res.json()["tasks"]
        assert len(tasks) == 3
        assert all(t["task_type"] == "plural_affixes" for t in tasks)

        # Assemble in scrambled order values [3, 1, 2]; expect Q1=order 1, Q2=order 2, Q3=order 3 in Variant 1
        # and distinct Variant 2 when shuffle_questions_in_variants=True with seed.
        payload_items = [
            {"task_id": tasks[2]["task_id"], "order": 1},
            {"task_id": tasks[0]["task_id"], "order": 2},
            {"task_id": tasks[1]["task_id"], "order": 3},
        ]
        asm_res = await client.post(
            "/api/v1/constructor/tests",
            json={
                "title": "Ordered Seed Test",
                "grade_level": 6,
                "task_items": payload_items,
                "generate_variants_count": 2,
                "shuffle_questions_in_variants": True,
                "seed": 123,
            },
        )
        assert asm_res.status_code == 201
        variants = asm_res.json()["variants"]
        assert [q["prompt"] for q in variants[0]["questions"]] == [
            tasks[2]["prompt_tt"],
            tasks[0]["prompt_tt"],
            tasks[1]["prompt_tt"],
        ]
        assert [q["prompt"] for q in variants[1]["questions"]] != [
            q["prompt"] for q in variants[0]["questions"]
        ]

    @pytest.mark.asyncio
    async def test_scan_task_base64_mime_and_data_uri_prefix(self, client: AsyncClient):
        bad_mime = await client.post(
            "/api/v1/constructor/scan-task-base64",
            json={
                "image_base64": base64.b64encode(b"fake").decode("ascii"),
                "mime_type": "application/pdf",
            },
        )
        assert bad_mime.status_code == 400

        raw_b64 = base64.b64encode(b"mock_png_payload").decode("ascii")
        ok_res = await client.post(
            "/api/v1/constructor/scan-task-base64",
            json={
                "image_base64": f"data:image/png;base64,{raw_b64}",
                "mime_type": "image/png",
                "grade_level": 5,
            },
        )
        assert ok_res.status_code == 200
        assert ok_res.json()["is_supported"] is True

    @pytest.mark.asyncio
    async def test_task_extractor_config_and_parsing_hardening(self, monkeypatch):
        prev_key = settings.YANDEX_API_KEY
        try:
            settings.YANDEX_API_KEY = ""
            with pytest.raises(ValueError, match="YANDEX_API_KEY"):
                get_task_extractor_service(use_mock=False)
        finally:
            settings.YANDEX_API_KEY = prev_key

        # Test YandexGPTClient response parsing with mocked HTTP response
        llm_payload = {
            "result": {
                "alternatives": [
                    {
                        "message": {
                            "text": json.dumps(
                                {
                                    "is_supported": False,
                                    "unsupported_reason": "Эссе соравы",
                                    "task_type": "custom",
                                    "confidence": None,
                                    "items": [
                                        {
                                            "prompt_tt": "Инша",
                                            "expected_answer": "КИТАП",
                                            "cell_count": "not-an-int",
                                            "grade_level": "bad",
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            }
        }

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, headers=None, json=None):
                req = httpx.Request("POST", url)
                return httpx.Response(200, json=llm_payload, request=req)

        monkeypatch.setattr("app.services.task_extractor.yandex_gpt.httpx.AsyncClient", FakeAsyncClient)
        gpt = YandexGPTClient(api_key="dummy-key", folder_id="dummy-folder")
        parsed = await gpt.classify_and_extract("OCR", registry.list_supported_types(), grade_hint=5)
        assert parsed.is_supported is False
        assert parsed.confidence == 0.9
        assert parsed.items[0].cell_count is None
        assert parsed.items[0].grade_level is None

        # Test TaskExtractorService grade_hint priority, 512-char prompt cap, and OCR alphabet filter
        class DummyOCR(BaseOCRClient):
            async def recognize_text(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
                return "OCR"

        class CustomLLM(BaseLLMTaskClassifier):
            def __init__(self, items):
                self._items = items

            async def classify_and_extract(self, ocr_text, supported_types, grade_hint=None):
                return ExtractedTaskResult(is_supported=True, task_type="custom", items=self._items)

        long_prompt = "Сүз " * 200
        svc = TaskExtractorService(
            ocr_client=DummyOCR(),
            llm_classifier=CustomLLM(
                [
                    ExtractedTaskSubItem(
                        task_type="custom",
                        prompt_tt=long_prompt,
                        expected_answer="КИТАП",
                        cell_count=8,
                        grade_level=9,
                    )
                ]
            ),
        )
        res = await svc.extract_from_image(b"img", grade_hint=5)
        assert res.is_supported is True
        assert res.task.grade_level == 5
        assert len(res.task.prompt_tt) <= 512

        bad_svc = TaskExtractorService(
            ocr_client=DummyOCR(),
            llm_classifier=CustomLLM(
                [
                    ExtractedTaskSubItem(
                        task_type="custom",
                        prompt_tt="Translate",
                        expected_answer="BOOK",
                        cell_count=8,
                        grade_level=7,
                    )
                ]
            ),
        )
        bad_res = await bad_svc.extract_from_image(b"img")
        assert bad_res.is_supported is False
