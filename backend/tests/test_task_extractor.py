import base64
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
from sqlalchemy import select

from app.main import app
from app.core.database import Base, get_db
from app.core.config import settings
from app.models.task import TaskBankModel
from app.services.task_extractor import (
    TaskExtractorService,
    get_task_extractor_service,
)
from app.services.task_extractor.mock_extractor import MockOCRClient, MockLLMTaskClassifier

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
    # Force mock mode in tests so no live cloud calls are made
    settings.TASK_EXTRACTOR_MOCK_MODE = True
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    app.dependency_overrides.clear()

@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestTaskExtractorUnit:
    """Unit tests for TaskExtractorService and linguistic/physical validations."""

    @pytest.mark.asyncio
    async def test_extract_case_inflection_task(self):
        service = TaskExtractorService(
            ocr_client=MockOCRClient(default_text="1-нче күнегү. Куегыз сүзне юнәлеш килешендә: китап ->"),
            llm_classifier=MockLLMTaskClassifier(),
        )
        fake_img = b"fake_image_bytes"
        res = await service.extract_from_image(fake_img, mime_type="image/jpeg", grade_hint=7)

        assert res.is_supported is True
        assert res.unsupported_reason is None
        assert res.task_type == "case_inflection"
        assert res.task is not None
        assert res.task.expected_answer == "КИТАПКА"
        assert res.task.cell_count >= len("КИТАПКА")
        assert res.task.cell_count <= 12
        assert res.task.grade_level == 7
        assert res.task.topic_tag == "case_dative"

    @pytest.mark.asyncio
    async def test_extract_unsupported_essay_task(self):
        service = TaskExtractorService(
            ocr_client=MockOCRClient(default_text="Җәйге яллар турында 100 сүздән торган инша языгыз."),
            llm_classifier=MockLLMTaskClassifier(),
        )
        fake_img = b"fake_image_bytes"
        res = await service.extract_from_image(fake_img, mime_type="image/png")

        assert res.is_supported is False
        assert res.unsupported_reason is not None
        assert "сочинения" in res.unsupported_reason or "бланками" in res.unsupported_reason
        assert res.task is None

    @pytest.mark.asyncio
    async def test_extract_multi_word_with_punctuation_gracefully_unsupported(self):
        class MultiWordLLMClassifier(MockLLMTaskClassifier):
            async def classify_and_extract(self, ocr_text, supported_types, grade_hint=None):
                from app.services.task_extractor.base import ExtractedTaskResult
                return ExtractedTaskResult(
                    is_supported=True,
                    task_type="custom",
                    prompt_tt="Сүзләрне табыгыз:",
                    expected_answer="Китап, укучы, яза.",
                    cell_count=8,
                    grade_level=7,
                    confidence=0.9,
                    raw_ocr_text=ocr_text,
                )

        service = TaskExtractorService(
            ocr_client=MockOCRClient(),
            llm_classifier=MultiWordLLMClassifier(),
        )
        res = await service.extract_from_image(b"fake_bytes", mime_type="image/jpeg")
        assert res.is_supported is False
        assert "несколько слов" in res.unsupported_reason or "знаки препинания" in res.unsupported_reason
        assert res.task is None

    @pytest.mark.asyncio
    async def test_extract_single_word_trailing_punctuation_stripped(self):
        class PunctuationLLMClassifier(MockLLMTaskClassifier):
            async def classify_and_extract(self, ocr_text, supported_types, grade_hint=None):
                from app.services.task_extractor.base import ExtractedTaskResult
                return ExtractedTaskResult(
                    is_supported=True,
                    task_type="case_inflection",
                    prompt_tt="Куегыз сүзне: китап ->",
                    expected_answer="«КИТАПКА».",
                    cell_count=8,
                    grade_level=7,
                    topic_tag="case_dative",
                    topic_name_tt="Юнәлеш",
                    confidence=0.95,
                    raw_ocr_text=ocr_text,
                )

        service = TaskExtractorService(
            ocr_client=MockOCRClient(),
            llm_classifier=PunctuationLLMClassifier(),
        )
        res = await service.extract_from_image(b"fake_bytes", mime_type="image/jpeg")
        assert res.is_supported is True
        assert res.task.expected_answer == "КИТАПКА"

    @pytest.mark.asyncio
    async def test_extract_multi_item_exercise_split_into_multiple_tasks(self):
        class MultiItemSplitLLMClassifier(MockLLMTaskClassifier):
            async def classify_and_extract(self, ocr_text, supported_types, grade_hint=None):
                from app.services.task_extractor.base import ExtractedTaskResult, ExtractedTaskSubItem
                return ExtractedTaskResult(
                    is_supported=True,
                    task_type="case_inflection",
                    items=[
                        ExtractedTaskSubItem(
                            task_type="case_inflection",
                            prompt_tt="Куегыз сүзне юнәлеш килешендә: китап ->",
                            expected_answer="КИТАПКА",
                            cell_count=8,
                            grade_level=7,
                            topic_tag="case_dative",
                            topic_name_tt="Юнәлеш килеше",
                        ),
                        ExtractedTaskSubItem(
                            task_type="case_inflection",
                            prompt_tt="Куегыз сүзне юнәлеш килешендә: укучы ->",
                            expected_answer="УКУЧЫГА",
                            cell_count=8,
                            grade_level=7,
                            topic_tag="case_dative",
                            topic_name_tt="Юнәлеш килеше",
                        ),
                        ExtractedTaskSubItem(
                            task_type="case_inflection",
                            prompt_tt="Куегыз сүзне юнәлеш килешендә: өстәл ->",
                            expected_answer="ӨСТӘЛГӘ",
                            cell_count=8,
                            grade_level=7,
                            topic_tag="case_dative",
                            topic_name_tt="Юнәлеш килеше",
                        ),
                    ],
                    confidence=0.96,
                    raw_ocr_text=ocr_text,
                )

        service = TaskExtractorService(
            ocr_client=MockOCRClient(),
            llm_classifier=MultiItemSplitLLMClassifier(),
        )
        res = await service.extract_from_image(b"fake_bytes", mime_type="image/jpeg")
        assert res.is_supported is True
        assert len(res.tasks) == 3
        assert res.tasks[0].expected_answer == "КИТАПКА"
        assert res.tasks[1].expected_answer == "УКУЧЫГА"
        assert res.tasks[2].expected_answer == "ӨСТӘЛГӘ"
        assert res.task.expected_answer == "КИТАПКА"

    @pytest.mark.asyncio
    async def test_extract_empty_image_raises_error(self):
        service = TaskExtractorService(
            ocr_client=MockOCRClient(),
            llm_classifier=MockLLMTaskClassifier(),
        )
        with pytest.raises(ValueError, match="empty"):
            await service.extract_from_image(b"", mime_type="image/jpeg")

    @pytest.mark.asyncio
    async def test_extract_oversized_image_raises_error(self):
        service = TaskExtractorService(
            ocr_client=MockOCRClient(),
            llm_classifier=MockLLMTaskClassifier(),
        )
        huge_bytes = b"0" * (11 * 1024 * 1024)
        with pytest.raises(ValueError, match="exceeds 10 MB limit"):
            await service.extract_from_image(huge_bytes, mime_type="image/jpeg")


class TestTaskExtractorAPI:
    """API integration tests for /constructor/scan-task and /constructor/scan-task-base64."""

    @pytest.mark.asyncio
    async def test_scan_task_multipart_and_save_to_bank(self, client: AsyncClient):
        # 1. Upload mock image with save_to_bank=True
        fake_image_content = b"fake_jpeg_content"
        files = {
            "file": ("exercise_photo.jpg", fake_image_content, "image/jpeg")
        }
        res = await client.post(
            "/api/v1/constructor/scan-task?save_to_bank=true&grade_level=7",
            files=files,
        )
        assert res.status_code == 200
        data = res.json()
        assert data["is_supported"] is True
        assert data["saved_to_bank"] is True
        assert data["task"] is not None
        task_id = data["task"]["task_id"]
        assert data["task"]["expected_answer"] == "КИТАПКА"

        # 2. Verify task was persisted into SQLite database
        async with TestSessionLocal() as session:
            stmt = select(TaskBankModel).where(TaskBankModel.task_id == task_id)
            result = await session.execute(stmt)
            saved_task = result.scalar_one_or_none()
            assert saved_task is not None
            assert saved_task.prompt_tt == data["task"]["prompt_tt"]
            assert saved_task.expected_answer == "КИТАПКА"
            assert saved_task.is_public_in_bank is True

    @pytest.mark.asyncio
    async def test_scan_task_base64_endpoint(self, client: AsyncClient):
        fake_image_b64 = base64.b64encode(b"fake_jpeg_image").decode("utf-8")
        payload = {
            "image_base64": fake_image_b64,
            "mime_type": "image/jpeg",
            "grade_level": 5,
            "save_to_bank": False,
        }
        res = await client.post("/api/v1/constructor/scan-task-base64", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["is_supported"] is True
        assert data["saved_to_bank"] is False
        assert data["task"]["expected_answer"] == "КИТАПКА"

    @pytest.mark.asyncio
    async def test_scan_task_invalid_content_type(self, client: AsyncClient):
        files = {
            "file": ("document.txt", b"plain text", "text/plain")
        }
        res = await client.post("/api/v1/constructor/scan-task", files=files)
        assert res.status_code == 400
        assert "Invalid file content type" in res.json()["detail"]

    @pytest.mark.asyncio
    async def test_scan_task_invalid_base64(self, client: AsyncClient):
        payload = {
            "image_base64": "not_valid_base64!!!",
            "mime_type": "image/jpeg",
        }
        res = await client.post("/api/v1/constructor/scan-task-base64", json=payload)
        assert res.status_code == 400


class TestYandexGPTJSONParsing:
    """Tests for YandexGPTClient._extract_json_from_text robustness."""

    def test_extract_json_with_trailing_commentary_and_extra_data(self):
        from app.services.task_extractor.yandex_gpt import YandexGPTClient
        client = YandexGPTClient(api_key="dummy", folder_id="dummy")

        # Case 1: Extra commentary after markdown code block
        text_with_extra = """```json
{
  "is_supported": true,
  "unsupported_reason": null,
  "task_type": "case_inflection",
  "prompt_tt": "Куегыз сүзне юнәлеш килешендә: китап ->",
  "expected_answer": "КИТАПКА",
  "cell_count": 8,
  "grade_level": 7,
  "topic_tag": "case_dative",
  "topic_name_tt": "Юнәлеш килеше",
  "confidence": 0.95
}
```
Пояснение: данное задание соответствует типу case_inflection, ответ КИТАПКА состоит из 7 букв."""

        parsed = client._extract_json_from_text(text_with_extra)
        assert parsed["is_supported"] is True
        assert parsed["expected_answer"] == "КИТАПКА"

        # Case 2: Raw JSON followed by extra curly braces and trailing text (extra data error cause)
        text_with_extra_braces = """{
  "is_supported": true,
  "task_type": "antonyms",
  "expected_answer": "ХАКЛЫК",
  "cell_count": 6
}

Дополнительная информация: { "status": "ok" }"""

        parsed2 = client._extract_json_from_text(text_with_extra_braces)
        assert parsed2["is_supported"] is True
        assert parsed2["expected_answer"] == "ХАКЛЫК"
