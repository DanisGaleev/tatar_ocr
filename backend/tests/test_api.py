import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

from app.main import app, lifespan
from app.core.database import Base, get_db
from app.models import TaskBankModel, AssembledTestModel, TeacherModel, ClassModel, StudentModel, SubmissionModel

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
    
    # Pre-seed sample task
    async with TestSessionLocal() as session:
        sample_task = TaskBankModel(
            task_id="tsk_test_01",
            prompt_tt="Куегыз сүзне юнәлеш килешендә: китап ->",
            expected_answer="КИТАПКА",
            cell_count=8,
            grade_level=7,
            topic_tag="case_dative",
            topic_name_tt="Юнәлеш килеше",
            task_type="case_inflection",
            is_public_in_bank=True,
        )
        session.add(sample_task)
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

class TestConstructorAndAssignmentsAPI:
    """Async API integration tests including adversarial error cases and schema invariants."""

    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_list_supported_task_types(self, client: AsyncClient):
        response = await client.get("/api/v1/constructor/task-types")
        assert response.status_code == 200
        data = response.json()
        assert "task_types" in data
        type_names = [t["task_type"] for t in data["task_types"]]
        assert "case_inflection" in type_names
        assert "plural_affixes" in type_names
        assert "antonyms" in type_names
        assert "translation" in type_names

    @pytest.mark.asyncio
    async def test_search_task_bank(self, client: AsyncClient):
        # 1. Search without filters returns seeded task
        res = await client.get("/api/v1/constructor/tasks")
        assert res.status_code == 200
        tasks = res.json()
        assert len(tasks) >= 1
        assert tasks[0]["task_id"] == "tsk_test_01"

        # 2. Search with matching query
        res = await client.get("/api/v1/constructor/tasks?query=китап")
        assert res.status_code == 200
        assert len(res.json()) >= 1

        # 3. Search with non-matching query returns empty list
        res = await client.get("/api/v1/constructor/tasks?query=NonExistentQuery")
        assert res.status_code == 200
        assert len(res.json()) == 0

    @pytest.mark.asyncio
    async def test_create_custom_task_success_and_persistence(self, client: AsyncClient):
        payload = {
            "prompt_tt": "Сүзгә күплек сан кушымчасын ялгагыз: дус ->",
            "expected_answer": "ДУСЛАР",
            "cell_count": 6,
            "grade_level": 5,
            "topic_tag": "plural_affixes",
            "topic_name_tt": "Күплек сан",
            "is_public_in_bank": True,
        }
        res = await client.post("/api/v1/constructor/tasks", json=payload)
        assert res.status_code == 201
        data = res.json()
        assert data["expected_answer"] == "ДУСЛАР"
        assert data["cell_count"] == 6
        assert data["task_id"].startswith("tsk_")

        # Verify it is now retrievable via bank search
        search_res = await client.get(f"/api/v1/constructor/tasks?query=ДУСЛАР")
        assert search_res.status_code == 200
        found = search_res.json()
        assert len(found) == 1
        assert found[0]["task_id"] == data["task_id"]

    @pytest.mark.asyncio
    async def test_create_custom_task_cell_overflow_rejected(self, client: AsyncClient):
        """Cell count smaller than answer length must be rejected with 422."""
        payload = {
            "prompt_tt": "Куегыз сүзне: китап ->",
            "expected_answer": "КИТАПКА",  # 7 chars
            "cell_count": 5,  # Only 5 cells provided -> Overflow!
            "grade_level": 7,
            "topic_tag": "case_dative",
            "topic_name_tt": "Юнәлеш килеше",
        }
        res = await client.post("/api/v1/constructor/tasks", json=payload)
        assert res.status_code == 422
        assert "exceeds specified cell count" in res.json()["detail"]

    @pytest.mark.asyncio
    async def test_generate_tasks_determinism_endpoint(self, client: AsyncClient):
        payload = {
            "task_type": "case_inflection",
            "count": 3,
            "seed": 4242,
            "grade_level": 7,
        }
        res1 = await client.post("/api/v1/constructor/generate", json=payload)
        res2 = await client.post("/api/v1/constructor/generate", json=payload)
        
        assert res1.status_code == 200
        assert res2.status_code == 200
        
        data1 = res1.json()
        data2 = res2.json()
        
        assert data1["count"] == 3
        # Invariant: identical seed returns identical prompts and expected answers
        assert [t["prompt_tt"] for t in data1["tasks"]] == [t["prompt_tt"] for t in data2["tasks"]]
        assert [t["expected_answer"] for t in data1["tasks"]] == [t["expected_answer"] for t in data2["tasks"]]

    @pytest.mark.asyncio
    async def test_generate_tasks_unknown_type_error(self, client: AsyncClient):
        payload = {
            "task_type": "quantum_physics",
            "count": 1,
        }
        res = await client.post("/api/v1/constructor/generate", json=payload)
        assert res.status_code == 400
        assert "Unknown task type" in res.json()["detail"]

    @pytest.mark.asyncio
    async def test_assemble_test_and_download_offline_bundle(self, client: AsyncClient):
        # 1. Assemble modular test automatically
        assemble_payload = {
            "title": "7 сыйныф. Исем килешләре контроль эше",
            "grade_level": 7,
            "generate_variants_count": 2,
            "shuffle_questions_in_variants": True,
            "seed": 999,
        }
        res = await client.post("/api/v1/constructor/tests", json=assemble_payload)
        assert res.status_code == 201
        bundle = res.json()
        
        assignment_id = bundle["assignment_id"]
        assert bundle["total_variants"] == 2
        assert len(bundle["variants"]) == 2
        assert len(bundle["variants"][0]["questions"]) == 8

        # 2. Retrieve offline bundle via assignments router
        bundle_res = await client.get(f"/api/v1/assignments/{assignment_id}/offline-bundle")
        assert bundle_res.status_code == 200
        retrieved_bundle = bundle_res.json()
        assert retrieved_bundle["assignment_id"] == assignment_id
        assert retrieved_bundle["total_variants"] == 2

        # 3. Retrieve specific single variant
        var1_res = await client.get(f"/api/v1/assignments/{assignment_id}/offline-bundle?variant=1")
        assert var1_res.status_code == 200
        var1_bundle = var1_res.json()
        assert var1_bundle["total_variants"] == 1
        assert var1_bundle["variants"][0]["variant_id"] == 1

    @pytest.mark.asyncio
    async def test_assemble_test_physical_a4_limit_overflow(self, client: AsyncClient):
        """Assembling a test with > 8 explicit questions must fail with 422."""
        items = [{"task_id": f"tsk_{i}", "order": i} for i in range(1, 10)]  # 9 questions!
        payload = {
            "title": "Too Long Test",
            "grade_level": 7,
            "generate_variants_count": 1,
            "task_items": items,
        }
        res = await client.post("/api/v1/constructor/tests", json=payload)
        assert res.status_code == 422
        assert "exceeds maximum A4 physical blank limit" in res.json()["detail"]

    @pytest.mark.asyncio
    async def test_offline_bundle_not_found(self, client: AsyncClient):
        res = await client.get("/api/v1/assignments/NON-EXISTENT-ID/offline-bundle")
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_web_ui_dashboard(self, client: AsyncClient):
        res_root = await client.get("/")
        assert res_root.status_code == 200
        assert "text/html" in res_root.headers["content-type"]
        assert "«Кара куян»" in res_root.text

        res_ui = await client.get("/ui")
        assert res_ui.status_code == 200
        assert "Deterministic Task Generator" in res_ui.text

    @pytest.mark.asyncio
    async def test_verify_answer_endpoint(self, client: AsyncClient):
        # 1. Correct answer
        res_ok = await client.post("/api/v1/constructor/verify-answer", json={
            "expected_answer": "КИТАПКА",
            "student_answer": "КИТАПКА",
            "cell_count": 8,
        })
        assert res_ok.status_code == 200
        data_ok = res_ok.json()
        assert data_ok["is_correct"] is True
        assert data_ok["points_earned"] == 1.0
        assert len(data_ok["cells"]) == 8

        # 2. Wrong answer (voicing mistake)
        res_wrong = await client.post("/api/v1/constructor/verify-answer", json={
            "expected_answer": "КИТАПКА",
            "student_answer": "КИТАПГА",
            "cell_count": 8,
        })
        assert res_wrong.status_code == 200
        data_wrong = res_wrong.json()
        assert data_wrong["is_correct"] is False
        assert data_wrong["points_earned"] == 0.0

