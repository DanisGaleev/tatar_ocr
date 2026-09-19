import logging
from typing import Optional, List, Dict
from app.services.task_extractor.base import BaseOCRClient, BaseLLMTaskClassifier, ExtractedTaskResult

logger = logging.getLogger(__name__)

class MockOCRClient(BaseOCRClient):
    def __init__(self, default_text: Optional[str] = None):
        self.default_text = default_text or "1-нче күнегү. Куегыз сүзне юнәлеш килешендә: китап ->"

    async def recognize_text(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        if image_bytes.startswith(b"TEXT:"):
            return image_bytes[5:].decode("utf-8", errors="replace")
        return self.default_text


class MockLLMTaskClassifier(BaseLLMTaskClassifier):
    async def classify_and_extract(
        self,
        ocr_text: str,
        supported_types: List[Dict[str, str]],
        grade_hint: Optional[int] = None,
    ) -> ExtractedTaskResult:
        text_lower = ocr_text.lower()

        # 1. Check for unsupported tasks (essay, long text, drawings)
        if any(w in text_lower for w in ["инша", "сочинение", "изложение", "рәсем ясагыз", "текст языгыз", "хикәя"]):
            return ExtractedTaskResult(
                is_supported=False,
                unsupported_reason="Задание требует написания развернутого сочинения/изложения, что не поддерживается бланками с ячейками (максимум 12 букв)",
                task_type=None,
                confidence=0.95,
                raw_ocr_text=ocr_text,
            )

        # 2. Check for case inflection
        if "килеш" in text_lower or "юнәлеш" in text_lower or "чыгыш" in text_lower:
            return ExtractedTaskResult(
                is_supported=True,
                task_type="case_inflection",
                prompt_tt="Куегыз сүзне юнәлеш килешендә: китап ->",
                expected_answer="КИТАПКА",
                cell_count=8,
                grade_level=grade_hint or 7,
                topic_tag="case_dative",
                topic_name_tt="Юнәлеш килеше",
                confidence=0.92,
                raw_ocr_text=ocr_text,
            )

        # 3. Check for plural affixes
        if "күплек" in text_lower or "сан" in text_lower or "кушымча" in text_lower:
            return ExtractedTaskResult(
                is_supported=True,
                task_type="plural_affixes",
                prompt_tt="Сүзгә күплек сан кушымчасын ялгагыз: бала ->",
                expected_answer="БАЛАЛАР",
                cell_count=8,
                grade_level=grade_hint or 5,
                topic_tag="plural_affixes",
                topic_name_tt="Күплек сан",
                confidence=0.90,
                raw_ocr_text=ocr_text,
            )

        # 4. Check for antonyms
        if "антоним" in text_lower or "кире" in text_lower:
            return ExtractedTaskResult(
                is_supported=True,
                task_type="antonyms",
                prompt_tt="Антонимны табыгыз: ялган ->",
                expected_answer="ХАКЛЫК",
                cell_count=6,
                grade_level=grade_hint or 6,
                topic_tag="antonyms_adjectives",
                topic_name_tt="Сыйфат антонимнары",
                confidence=0.91,
                raw_ocr_text=ocr_text,
            )

        # 5. Default fallback to custom task
        return ExtractedTaskResult(
            is_supported=True,
            task_type="custom",
            prompt_tt=ocr_text.strip()[:200],
            expected_answer="ҮРНӘК",
            cell_count=8,
            grade_level=grade_hint or 7,
            topic_tag="custom_task",
            topic_name_tt="Үзләштерү биреме",
            confidence=0.80,
            raw_ocr_text=ocr_text,
        )
