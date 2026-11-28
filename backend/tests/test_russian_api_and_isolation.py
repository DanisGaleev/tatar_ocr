import base64
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db
from app.core.config import settings
from app.models.task import TaskBankModel
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
    settings.TASK_EXTRACTOR_MOCK_MODE = True
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        tt_task = TaskBankModel(
            task_id="tsk_tt_seed",
            language="tt",
            prompt_tt="Куегыз сүзне юнәлеш килешендә: китап ->",
            expected_answer="КИТАПКА",
            cell_count=8,
            grade_level=7,
            topic_tag="case_dative",
            topic_name_tt="Юнәлеш килеше",
            task_type="case_inflection",
            is_public_in_bank=True,
        )
        ru_task = TaskBankModel(
            task_id="tsk_ru_seed",
            language="ru",
            prompt_tt="Вставьте букву и запишите слово целиком (ЖИ–ШИ): маш_на ->",
            expected_answer="МАШИНА",
            cell_count=8,
            grade_level=1,
            topic_tag="ru_orthogram_combinations",
            topic_name_tt="Правописание ЖИ–ШИ, ЧА–ЩА, ЧУ–ЩУ",
            task_type="ru_orthogram_combinations",
            is_public_in_bank=True,
        )
        session.add_all([tt_task, ru_task])
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


class TestRussianFourModesAndParallelIsolation:
    """End-to-end API tests covering all 4 Russian task modes and tt/ru isolation."""

    @pytest.mark.asyncio
    async def test_mode_1_deterministic_generation_and_cross_language_guard(self, client: AsyncClient):
        types_ru = await client.get("/api/v1/constructor/task-types?language=ru")
        assert types_ru.status_code == 200
        ru_list = types_ru.json()["task_types"]
        assert len(ru_list) == 14
        assert all(t["language"] == "ru" for t in ru_list)

        types_tt = await client.get("/api/v1/constructor/task-types?language=tt")
        assert types_tt.status_code == 200
        tt_list = types_tt.json()["task_types"]
        assert len(tt_list) == 24
        assert all(t["language"] == "tt" for t in tt_list)

        # Deterministic generation in Russian + save_to_bank
        gen_res = await client.post(
            "/api/v1/constructor/generate",
            json={
                "task_type": "ru_orthogram_combinations",
                "language": "ru",
                "count": 2,
                "seed": 101,
                "grade_level": 2,
                "custom_stems": ["снежинка", "чайник"],
                "save_to_bank": True,
            },
        )
        assert gen_res.status_code == 200
        data = gen_res.json()
        assert data["language"] == "ru"
        assert data["count"] == 2
        assert all(t["language"] == "ru" for t in data["tasks"])
        assert {t["expected_answer"] for t in data["tasks"]}.issubset({"СНЕЖИНКА", "ЧАЙНИК"})

        # Cross-language mismatch must fail with 400
        bad_gen = await client.post(
            "/api/v1/constructor/generate",
            json={"task_type": "case_inflection", "language": "ru", "count": 1},
        )
        assert bad_gen.status_code == 400

    @pytest.mark.asyncio
    async def test_mode_2_manual_russian_task_creation_and_isolation(self, client: AsyncClient):
        res = await client.post(
            "/api/v1/constructor/tasks",
            json={
                "language": "ru",
                "prompt": "Вставьте безударную гласную (проверочное: лес): л_сной ->",
                "expected_answer": "Е",
                "cell_count": 1,
                "grade_level": 2,
                "topic_tag": "ru_unstressed_vowel",
                "topic_name": "Безударная гласная",
                "is_public_in_bank": True,
            },
        )
        assert res.status_code == 201
        created = res.json()
        assert created["language"] == "ru"
        assert created["expected_answer"] == "Е"
        assert created["grade_level"] == 2

        # Must appear in language=ru&scope=my_tasks, NOT in language=tt
        ru_my = await client.get("/api/v1/constructor/tasks?language=ru&scope=my_tasks")
        assert ru_my.status_code == 200
        assert any(t["task_id"] == created["task_id"] for t in ru_my.json())

        tt_my = await client.get("/api/v1/constructor/tasks?language=tt&scope=my_tasks")
        assert tt_my.status_code == 200
        assert tt_my.json() == []

        # Tatar-only letter in Russian custom task must be rejected with 422
        bad_res = await client.post(
            "/api/v1/constructor/tasks",
            json={
                "language": "ru",
                "prompt_tt": "Тест татарской буквы:",
                "expected_answer": "ӘНИ",
                "cell_count": 4,
                "grade_level": 2,
                "topic_tag": "custom",
                "topic_name_tt": "Тест",
            },
        )
        assert bad_res.status_code == 422

    @pytest.mark.asyncio
    async def test_mode_3_photo_scan_russian_multipart_and_base64(self, client: AsyncClient):
        # Multipart upload with TEXT: payload for mock OCR
        text_payload = b"TEXT:2 \xd0\xba\xd0\xbb\xd0\xb0\xd1\x81\xd1\x81. \xd0\x92\xd1\x81\xd1\x82\xd0\xb0\xd0\xb2\xd1\x8c\xd1\x82\xd0\xb5 \xd0\xb1\xd0\xb5\xd0\xb7\xd1\x83\xd0\xb4\xd0\xb0\xd1\x80\xd0\xbd\xd1\x83\xd1\x8e \xd0\xb3\xd0\xbb\xd0\xb0\xd1\x81\xd0\xbd\xd1\x83\xd1\x8e: \xd0\xbb_\xd1\x81\xd0\xbd\xd0\xbe\xd0\xb9"
        files = {"file": ("ru_ex.jpg", text_payload, "image/jpeg")}
        scan_res = await client.post(
            "/api/v1/constructor/scan-task?language=ru&save_to_bank=true&grade_level=2",
            files=files,
        )
        assert scan_res.status_code == 200
        scan_data = scan_res.json()
        assert scan_data["is_supported"] is True
        assert scan_data["language"] == "ru"
        assert scan_data["saved_to_bank"] is True
        assert scan_data["task"]["language"] == "ru"
        assert scan_data["task"]["expected_answer"] == "Е"

        # Base64 scan with Russian antonym exercise
        b64_img = base64.b64encode("Подберите антоним к слову: день".encode("utf-8")).decode("ascii")
        # Prefix with TEXT: so MockOCRClient reads the embedded text
        b64_text = base64.b64encode(b"TEXT:" + "Подберите антоним к слову: день".encode("utf-8")).decode("ascii")
        b64_res = await client.post(
            "/api/v1/constructor/scan-task-base64",
            json={
                "image_base64": b64_text,
                "mime_type": "image/jpeg",
                "language": "ru",
                "grade_level": 2,
                "save_to_bank": False,
            },
        )
        assert b64_res.status_code == 200
        b64_data = b64_res.json()
        assert b64_data["is_supported"] is True
        assert b64_data["language"] == "ru"
        assert b64_data["task"]["expected_answer"] == "НОЧЬ"

        # Unsupported essay exercise in Russian
        essay_files = {
            "file": ("essay.png", b"TEXT:" + "Напишите сочинение о лете.".encode("utf-8"), "image/png")
        }
        essay_res = await client.post("/api/v1/constructor/scan-task?language=ru", files=essay_files)
        assert essay_res.status_code == 200
        assert essay_res.json()["is_supported"] is False

    @pytest.mark.asyncio
    async def test_mode_4_ready_tasks_and_tests_from_system_plus_pdf_and_manifest(self, client: AsyncClient):
        # 1. Search task bank isolated by language
        ru_bank = await client.get("/api/v1/constructor/tasks?language=ru")
        assert ru_bank.status_code == 200
        ru_ids = {t["task_id"] for t in ru_bank.json()}
        assert "tsk_ru_seed" in ru_ids
        assert "tsk_tt_seed" not in ru_ids

        # 2. Mixing tt and ru tasks into a Russian test must be rejected (422)
        mix_res = await client.post(
            "/api/v1/constructor/tests",
            json={
                "title": "Смешанный тест",
                "language": "ru",
                "grade_level": 3,
                "task_items": [
                    {"task_id": "tsk_ru_seed", "order": 1},
                    {"task_id": "tsk_tt_seed", "order": 2},
                ],
            },
        )
        assert mix_res.status_code == 422

        # 3. Assemble auto Russian test
        asm_res = await client.post(
            "/api/v1/constructor/tests",
            json={
                "title": "Русский язык. 3 класс. Итоговый тест",
                "language": "ru",
                "grade_level": 3,
                "generate_variants_count": 2,
                "seed": 777,
            },
        )
        assert asm_res.status_code == 201
        bundle = asm_res.json()
        aid = bundle["assignment_id"]
        assert aid.startswith("RUS-")
        assert bundle["language"] == "ru"
        assert bundle["alphabet_mask"] == "russian_33"
        assert len(bundle["variants"][0]["questions"]) == 8

        # 4. Ready tests & assignments listing filter by language
        ready_ru = await client.get("/api/v1/constructor/ready-tests?language=ru")
        assert ready_ru.status_code == 200
        assert any(r["test_id"] == aid and r["language"] == "ru" for r in ready_ru.json())

        ready_tt = await client.get("/api/v1/constructor/ready-tests?language=tt")
        assert ready_tt.status_code == 200
        assert not any(r["test_id"] == aid for r in ready_tt.json())

        # 5. Offline bundle and printable PDF blank for Russian test
        off_res = await client.get(f"/api/v1/assignments/{aid}/offline-bundle")
        assert off_res.status_code == 200
        assert off_res.json()["language"] == "ru"
        assert off_res.json()["alphabet_mask"] == "russian_33"

        pdf_res = await client.get(f"/api/v1/assignments/{aid}/blank.pdf?variant=1")
        assert pdf_res.status_code == 200
        assert pdf_res.headers["content-type"] == "application/pdf"
        assert pdf_res.content.startswith(b"%PDF")

        # 6. Verify-answer endpoint with Russian homoglyph Y -> У
        ver_res = await client.post(
            "/api/v1/constructor/verify-answer",
            json={
                "expected_answer": "УТРО",
                "student_answer": "YTPO",
                "cell_count": 8,
                "language": "ru",
            },
        )
        assert ver_res.status_code == 200
        assert ver_res.json()["is_correct"] is True

        # 7. Model manifest includes both 39-class Tatar and 33-class Russian mask
        man_res = await client.get("/api/v1/model/manifest")
        assert man_res.status_code == 200
        manifest = man_res.json()
        assert manifest["num_classes"] == 39
        assert len(manifest["language_alphabets"]["tt"]) == 39
        assert len(manifest["language_alphabets"]["ru"]) == 33
        assert len(manifest["language_masks"]["ru"]) == 33
        for tatar_ch in ("Ә", "Ө", "Ү", "Җ", "Ң", "Һ"):
            assert tatar_ch not in manifest["language_alphabets"]["ru"]
