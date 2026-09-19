import uuid
import logging
from typing import Optional, List, Dict
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.generators.registry import registry
from app.generators.phonetics import clean_word
from app.models.task import TaskBankModel
from app.schemas.task import TaskBankItem, ScanTaskResponse
from app.services.task_extractor.base import (
    BaseOCRClient,
    BaseLLMTaskClassifier,
    ExtractedTaskResult,
)
from app.services.task_extractor.yandex_ocr import YandexOCRClient
from app.services.task_extractor.yandex_gpt import YandexGPTClient
from app.services.task_extractor.mock_extractor import MockOCRClient, MockLLMTaskClassifier

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

class TaskExtractorService:
    def __init__(
        self,
        ocr_client: BaseOCRClient,
        llm_classifier: BaseLLMTaskClassifier,
    ):
        self.ocr_client = ocr_client
        self.llm_classifier = llm_classifier

    async def extract_from_image(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        grade_hint: Optional[int] = None,
        save_to_bank: bool = False,
        db: Optional[AsyncSession] = None,
    ) -> ScanTaskResponse:
        # 1. Validation
        if not image_bytes:
            raise ValueError("Uploaded image file is empty.")
        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(f"Image size ({len(image_bytes)} bytes) exceeds 10 MB limit.")

        # 2. OCR Stage
        logger.info("Executing OCR on %d bytes (mime: %s)...", len(image_bytes), mime_type)
        raw_ocr_text = await self.ocr_client.recognize_text(image_bytes, mime_type=mime_type)
        if not raw_ocr_text or not raw_ocr_text.strip():
            return ScanTaskResponse(
                is_supported=False,
                unsupported_reason="Текст на изображении не обнаружен или не удалось распознать (OCR вернул пустой результат).",
                raw_ocr_text="",
                confidence=0.0,
                saved_to_bank=False,
            )

        # 3. LLM Classification & Structuring Stage
        supported_types = registry.list_supported_types()
        logger.info("Classifying task with LLM against %d supported types...", len(supported_types))
        extracted: ExtractedTaskResult = await self.llm_classifier.classify_and_extract(
            ocr_text=raw_ocr_text,
            supported_types=supported_types,
            grade_hint=grade_hint,
        )

        # 4. Check if LLM marked as unsupported or returned no items
        candidate_items = extracted.items
        if not candidate_items and extracted.expected_answer and extracted.prompt_tt:
            from app.services.task_extractor.base import ExtractedTaskSubItem
            candidate_items = [
                ExtractedTaskSubItem(
                    task_type=extracted.task_type or "custom",
                    prompt_tt=extracted.prompt_tt,
                    expected_answer=extracted.expected_answer,
                    cell_count=extracted.cell_count,
                    grade_level=extracted.grade_level,
                    topic_tag=extracted.topic_tag,
                    topic_name_tt=extracted.topic_name_tt,
                )
            ]

        if not extracted.is_supported or not candidate_items:
            return ScanTaskResponse(
                is_supported=False,
                unsupported_reason=extracted.unsupported_reason or "Данный тип задания не поддерживается бланковой системой.",
                raw_ocr_text=raw_ocr_text,
                task_type=extracted.task_type,
                confidence=extracted.confidence,
                saved_to_bank=False,
            )

        # 5. Process and validate all sub-tasks
        valid_tasks: List[TaskBankItem] = []
        validation_errors: List[str] = []

        for idx, item in enumerate(candidate_items, start=1):
            raw_ans = (item.expected_answer or "").strip().strip(".,;:!?\"'«»()[]-")
            if not raw_ans:
                continue

            # Skip or reject if answer contains multiple words or punctuation
            if any(char in raw_ans for char in ",;:!?\"'«»()[]") or " " in raw_ans:
                validation_errors.append(f"Пункт {idx}: ответ '{raw_ans}' содержит несколько слов или знаки препинания.")
                continue

            # Tatar Cyrillic validation
            try:
                clean_ans = clean_word(raw_ans).upper()
            except Exception as err:
                validation_errors.append(f"Пункт {idx}: ответ '{raw_ans}' содержит недопустимые символы: {err}")
                continue

            # Physical limit: max 12 cells on A4 template
            if len(clean_ans) > 12:
                validation_errors.append(f"Пункт {idx}: ответ '{clean_ans}' ({len(clean_ans)} букв) превышает 12 ячеек.")
                continue

            cell_count = item.cell_count or max(8, len(clean_ans))
            cell_count = max(1, min(12, cell_count))
            if len(clean_ans) > cell_count:
                cell_count = len(clean_ans)

            grade_level = item.grade_level or grade_hint or extracted.grade_level or 7
            grade_level = max(5, min(9, grade_level))

            topic_tag = item.topic_tag or extracted.topic_tag or "custom"
            topic_name_tt = item.topic_name_tt or extracted.topic_name_tt or "Үзләштерү биреме"
            prompt_tt = (item.prompt_tt or raw_ocr_text).strip()
            item_task_type = item.task_type or extracted.task_type or "custom"

            task_id = f"tsk_{uuid.uuid4().hex[:8]}"

            task_item = TaskBankItem(
                task_id=task_id,
                prompt_tt=prompt_tt,
                expected_answer=clean_ans,
                cell_count=cell_count,
                grade_level=grade_level,
                topic_tag=topic_tag,
                topic_name_tt=topic_name_tt,
                is_public_in_bank=save_to_bank,
            )
            valid_tasks.append(task_item)

            if save_to_bank and db is not None:
                db_task = TaskBankModel(
                    task_id=task_id,
                    prompt_tt=prompt_tt,
                    expected_answer=clean_ans,
                    cell_count=cell_count,
                    grade_level=grade_level,
                    topic_tag=topic_tag,
                    topic_name_tt=topic_name_tt,
                    task_type=item_task_type,
                    is_public_in_bank=True,
                )
                db.add(db_task)

        if save_to_bank and db is not None and valid_tasks:
            await db.commit()
            logger.info("Persisted %d scanned sub-tasks into task bank.", len(valid_tasks))

        if not valid_tasks:
            reasons = "; ".join(validation_errors) if validation_errors else "Не удалось сформировать ни одного подходящего задания."
            return ScanTaskResponse(
                is_supported=False,
                unsupported_reason=f"Ни один пункт упражнения не подошел для бланка А4: {reasons}",
                raw_ocr_text=raw_ocr_text,
                task_type=extracted.task_type,
                confidence=extracted.confidence,
                saved_to_bank=False,
            )

        return ScanTaskResponse(
            is_supported=True,
            unsupported_reason=None,
            task_type=valid_tasks[0].task_type if hasattr(valid_tasks[0], "task_type") else extracted.task_type,
            task=valid_tasks[0],
            tasks=valid_tasks,
            raw_ocr_text=raw_ocr_text,
            confidence=extracted.confidence,
            saved_to_bank=save_to_bank and (db is not None),
        )


def get_task_extractor_service(use_mock: Optional[bool] = None) -> TaskExtractorService:
    """
    Factory creating TaskExtractorService. Uses mock if specified or if API key is missing.
    """
    mock_mode = use_mock if use_mock is not None else settings.TASK_EXTRACTOR_MOCK_MODE
    if not settings.YANDEX_API_KEY:
        mock_mode = True

    if mock_mode:
        ocr_client = MockOCRClient()
        llm_classifier = MockLLMTaskClassifier()
    else:
        ocr_client = YandexOCRClient()
        llm_classifier = YandexGPTClient()

    return TaskExtractorService(ocr_client=ocr_client, llm_classifier=llm_classifier)
