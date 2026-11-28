import logging
from typing import Optional, List, Dict
from app.services.task_extractor.base import BaseOCRClient, BaseLLMTaskClassifier, ExtractedTaskResult

logger = logging.getLogger(__name__)

class MockOCRClient(BaseOCRClient):
    def __init__(self, default_text: Optional[str] = None):
        self.default_text = default_text

    async def recognize_text(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        language: str = "tt",
    ) -> str:
        if image_bytes.startswith(b"TEXT:"):
            return image_bytes[5:].decode("utf-8", errors="replace")
        if self.default_text is not None:
            return self.default_text
        if language == "ru":
            return "Упражнение 1. Вставьте безударную гласную (проверочное слово: лес): л_сной ->"
        return "1-нче күнегү. Куегыз сүзне юнәлеш килешендә: китап ->"


class MockLLMTaskClassifier(BaseLLMTaskClassifier):
    async def classify_and_extract(
        self,
        ocr_text: str,
        supported_types: List[Dict[str, str]],
        grade_hint: Optional[int] = None,
        language: str = "tt",
    ) -> ExtractedTaskResult:
        text_lower = ocr_text.lower()

        # 1. Check for unsupported tasks (essay, long text, drawings)
        if any(w in text_lower for w in ["инша", "сочинение", "изложение", "рәсем ясагыз", "текст языгыз", "хикәя", "нарисуйте"]):
            return ExtractedTaskResult(
                is_supported=False,
                unsupported_reason="Задание требует написания развернутого сочинения/изложения, что не поддерживается бланками с ячейками (максимум 12 букв)",
                task_type=None,
                confidence=0.95,
                raw_ocr_text=ocr_text,
            )

        if language == "ru":
            if "безударн" in text_lower or "л_с" in text_lower:
                return ExtractedTaskResult(
                    is_supported=True,
                    task_type="ru_unstressed_vowel",
                    prompt_tt="Вставьте безударную гласную (проверочное слово: лес): л_сной ->",
                    expected_answer="Е",
                    cell_count=1,
                    grade_level=grade_hint or 2,
                    topic_tag="ru_unstressed_vowel",
                    topic_name_tt="Безударная гласная в корне слова",
                    confidence=0.94,
                    raw_ocr_text=ocr_text,
                )
            if any(w in text_lower for w in ["жи", "ши", "ча", "ща", "чу", "щу", "маш_на"]):
                return ExtractedTaskResult(
                    is_supported=True,
                    task_type="ru_orthogram_combinations",
                    prompt_tt="Вставьте букву и запишите слово целиком (ЖИ–ШИ): маш_на ->",
                    expected_answer="МАШИНА",
                    cell_count=8,
                    grade_level=grade_hint or 1,
                    topic_tag="ru_orthogram_combinations",
                    topic_name_tt="Правописание ЖИ–ШИ, ЧА–ЩА, ЧУ–ЩУ",
                    confidence=0.93,
                    raw_ocr_text=ocr_text,
                )
            if "род" in text_lower:
                return ExtractedTaskResult(
                    is_supported=True,
                    task_type="ru_noun_gender",
                    prompt_tt="Определите род существительного (М — мужской, Ж — женский, С — средний): солнце ->",
                    expected_answer="С",
                    cell_count=1,
                    grade_level=grade_hint or 3,
                    topic_tag="ru_noun_gender",
                    topic_name_tt="Род имён существительных",
                    confidence=0.92,
                    raw_ocr_text=ocr_text,
                )
            if "антоним" in text_lower:
                return ExtractedTaskResult(
                    is_supported=True,
                    task_type="ru_antonyms",
                    prompt_tt="Подберите антоним (противоположное по смыслу слово): день ->",
                    expected_answer="НОЧЬ",
                    cell_count=8,
                    grade_level=grade_hint or 2,
                    topic_tag="ru_antonyms",
                    topic_name_tt="Антонимы (слова с противоположным значением)",
                    confidence=0.91,
                    raw_ocr_text=ocr_text,
                )
            return ExtractedTaskResult(
                is_supported=True,
                task_type="custom",
                prompt_tt=ocr_text.strip()[:200],
                expected_answer="СЛОВО",
                cell_count=8,
                grade_level=grade_hint or 3,
                topic_tag="custom_task",
                topic_name_tt="Задание учителя",
                confidence=0.80,
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
