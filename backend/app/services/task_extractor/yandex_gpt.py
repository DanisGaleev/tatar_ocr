import json
import logging
import re
from typing import Optional, List, Dict, Any
import httpx

from app.core.config import settings
from app.services.task_extractor.base import (
    BaseLLMTaskClassifier,
    ExtractedTaskResult,
    ExtractedTaskSubItem,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """Ты — интеллектуальный ассистент образовательной платформы «Дәресханә» (Tatar OCR).
Твоя задача — проанализировать распознанный OCR-текст школьного задания по татарскому языку (из учебника, сборника упражнений или теста) и структурировать его в строгий JSON-формат для конструктора заданий.

### НАШИ ПОДДЕРЖИВАЕМЫЕ ТИПЫ ЗАДАНИЙ:
{supported_types_description}

Также поддерживается тип "custom" — если это другое задание по татарскому языку, где ответом является одно слово или краткий код до 12 символов.

### КРИТИЧЕСКИЕ ПРАВИЛА ТАТАРСКОГО ЯЗЫКА И ПЕРЕВОДА:
1. Платформа «Дәресханә» создана ИСКЛЮЧИТЕЛЬНО ДЛЯ ИЗУЧЕНИЯ ТАТАРСКОГО ЯЗЫКА.
2. СТРОГИЙ ЗАПРЕТ АНГЛИЙСКОГО ЯЗЫКА: НИКАКИХ АНГЛИЙСКИХ СЛОВ И ЛАТИНИЦЫ!
   Если в задании написано "Переведи слова", "Переведите" (например: "Переведи данные слова: Книга, ученик, пишет" или "Тәрҗемә итегез"):
   - ПЕРЕВОД ВСЕГДА ВЫПОЛНЯЕТСЯ НА ТАТАРСКИЙ ЯЗЫК!
     Примеры:
     * "Книга" -> "КИТАП"
     * "Ученик" -> "УКУЧЫ"
     * "Пишет" -> "ЯЗА"
     * "Учитель" -> "УКЫТУЧЫ"
     * "Школа" -> "МӘКТӘП"
   - НИ В КОЕМ СЛУЧАЕ НЕ ПЕРЕВОДИ НА АНГЛИЙСКИЙ (КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать "BOOK", "STUDENT", "WRITES")!
   - Для таких заданий используй:
     * task_type: "translation"
     * topic_tag: "vocabulary_translation"
     * topic_name_tt: "Сүзлек байлыгы (тәрҗемә)"
     * prompt_tt: "Тәрҗемә итегез (книга): ->"
3. Ответ (`expected_answer`) ВСЕГДА ДОЛЖЕН БЫТЬ НА ТАТАРСКОМ ЯЗЫКЕ (татарская кириллица: А-Я, Ә, Ө, Ү, Җ, Ң, Һ).
4. ИГНОРИРУЙ МУСОРНЫЙ OCR-ТЕКСТ: баллы за задание (например: "(3 балл)", "балл"), номера страниц ("— 18 - 17", "14", "— 16 —"), колонтитулы и артефакты сканирования. Извлекай только само учебное упражнение.

### ПРАВИЛА РАЗБИЕНИЯ НА ПОДЗАДАНИЯ:
1. Если упражнение в учебнике содержит несколько пунктов, примеров или слов (например: "1) китап, 2) укучы, 3) яза" или "Переведи слова: книга, ученик, пишет"):
   ОБЯЗАТЕЛЬНО РАЗБЕЙ ЕГО НА ОТДЕЛЬНЫЕ САМОСТОЯТЕЛЬНЫЕ ЗАДАНИЯ в массиве `items`!
   Каждое задание в `items` должно содержать:
   - `prompt_tt`: полную понятную формулировку со словом (например: "Тәрҗемә итегез (книга): ->")
   - `expected_answer`: РОВНО ОДНО ТАТАРСКОЕ СЛОВО-ОТВЕТ (например: "КИТАП")
   - `cell_count`: количество ячеек (от 1 до 12, по длине слова или 8)
   - `task_type`: подходящий тип (например: "translation")
   - `topic_tag` и `topic_name_tt`
2. Если в упражнении есть и развернутые части (сочинение, свободный текст), и краткие части (вставить букву, раскрыть скобки, поставить слово в падеж, перевести слово) — ИЗВЛЕКИ ВСЕ КРАТКИЕ ЧАСТИ в массив `items`!
3. Ответ (`expected_answer`) в КАЖДОМ задании ДОЛЖЕН быть РОВНО ОДНИМ СЛОВОМ длиной от 1 до 12 букв (только татарская кириллица: Ә, Ө, Ү, Җ, Ң, Һ). Без запятых, пробелов и знаков препинания.
4. Если во всем тексте нет ни одного пункта, который можно сформулировать как задание с ответом в одно слово до 12 букв — только тогда установи `is_supported: false` и подробно укажи причину в `unsupported_reason`.

### ФОРМАТ ОТВЕТА (ТОЛЬКО ЧИСТЫЙ JSON):
```json
{{
  "is_supported": true,
  "unsupported_reason": null,
  "task_type": "case_inflection",
  "items": [
    {{
      "prompt_tt": "Куегыз сүзне юнәлеш килешендә: китап ->",
      "expected_answer": "КИТАПКА",
      "cell_count": 8,
      "grade_level": 7,
      "task_type": "case_inflection",
      "topic_tag": "case_dative",
      "topic_name_tt": "Юнәлеш килеше"
    }},
    {{
      "prompt_tt": "Куегыз сүзне юнәлеш килешендә: укучы ->",
      "expected_answer": "УКУЧЫГА",
      "cell_count": 8,
      "grade_level": 7,
      "task_type": "case_inflection",
      "topic_tag": "case_dative",
      "topic_name_tt": "Юнәлеш килеше"
    }}
  ],
  "confidence": 0.95
}}
```

Если не поддерживается вообще (нет ни одной краткой части):
```json
{{
  "is_supported": false,
  "unsupported_reason": "Задание требует написания сочинения и не содержит вопросов с кратким ответом в одно слово",
  "task_type": null,
  "items": [],
  "confidence": 0.90
}}
```

Отвечай ИСКЛЮЧИТЕЛЬНО валидным JSON без лишних пояснений."""

class YandexGPTClient(BaseLLMTaskClassifier):
    def __init__(
        self,
        api_key: Optional[str] = None,
        folder_id: Optional[str] = None,
        model_uri: Optional[str] = None,
        gpt_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or settings.YANDEX_API_KEY
        self.folder_id = folder_id or settings.YANDEX_FOLDER_ID
        self.gpt_url = gpt_url or settings.YANDEX_GPT_URL
        self.timeout = timeout
        
        # Determine model URI
        if model_uri:
            self.model_uri = model_uri
        elif settings.YANDEX_GPT_MODEL_URI:
            self.model_uri = settings.YANDEX_GPT_MODEL_URI
        elif self.folder_id:
            self.model_uri = f"gpt://{self.folder_id}/yandexgpt-lite/latest"
        else:
            self.model_uri = ""

    def _build_supported_types_desc(self, supported_types: List[Dict[str, str]]) -> str:
        lines = []
        for t in supported_types:
            task_type = t.get("task_type", "")
            topic_tag = t.get("topic_tag", "")
            topic_name_tt = t.get("topic_name_tt", "")
            lines.append(f"- {task_type} (тема: {topic_name_tt} / {topic_tag})")
        return "\n".join(lines)

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()

        # 1. First attempt: match markdown code block ```json ... ```
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if code_block_match:
            candidate = code_block_match.group(1).strip()
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # 2. Second attempt: JSONDecoder.raw_decode from first '{' (ignores any trailing commentary)
        start_idx = cleaned.find("{")
        if start_idx != -1:
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(cleaned, idx=start_idx)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

            # 3. Third attempt: extract slice between { and } and sanitize trailing commas
            end_idx = cleaned.rfind("}")
            if end_idx > start_idx:
                chunk = cleaned[start_idx : end_idx + 1]
                sanitized = re.sub(r",\s*([\]}])", r"\1", chunk)
                try:
                    obj = json.loads(sanitized)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass

        # If all fail, log raw text and raise clear error
        logger.error("Failed to parse JSON from YandexGPT output: %s", text)
        raise ValueError(f"Could not parse valid JSON from LLM response: {text[:200]}")

    async def classify_and_extract(
        self,
        ocr_text: str,
        supported_types: List[Dict[str, str]],
        grade_hint: Optional[int] = None,
    ) -> ExtractedTaskResult:
        if not self.api_key:
            raise ValueError("Yandex API key is not configured for YandexGPT service.")
        if not self.model_uri:
            raise ValueError("Yandex model URI or folder ID is not configured for YandexGPT service.")

        types_desc = self._build_supported_types_desc(supported_types)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(supported_types_description=types_desc)

        user_content = f"Распознанный текст задания с фото:\n\"\"\"\n{ocr_text}\n\"\"\""
        if grade_hint:
            user_content += f"\nПодсказка класса от учителя: {grade_hint} класс"

        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.folder_id:
            headers["x-folder-id"] = self.folder_id

        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": 0.1,
                "maxTokens": 1200,
            },
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_content},
            ],
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.gpt_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                err_detail = exc.response.text or str(exc)
                logger.error("YandexGPT HTTP error %s: %s", exc.response.status_code, err_detail)
                raise RuntimeError(f"YandexGPT HTTP {exc.response.status_code}: {err_detail}") from exc
            except httpx.TimeoutException as exc:
                logger.error("YandexGPT request timed out after %s seconds", self.timeout)
                raise RuntimeError(f"YandexGPT timeout: запрос превысил лимит ожидания ({self.timeout}с). Попробуйте еще раз.") from exc
            except Exception as exc:
                err_msg = str(exc) or repr(exc) or type(exc).__name__
                logger.error("YandexGPT connection error: %s (%s)", type(exc).__name__, err_msg)
                raise RuntimeError(f"YandexGPT {type(exc).__name__}: {err_msg}") from exc

        alternatives = data.get("result", {}).get("alternatives", [])
        if not alternatives:
            raise RuntimeError("YandexGPT returned no completion alternatives.")

        raw_llm_text = alternatives[0].get("message", {}).get("text", "")
        logger.debug("Raw YandexGPT response: %s", raw_llm_text)

        parsed = self._extract_json_from_text(raw_llm_text)

        is_supported = bool(parsed.get("is_supported", False))
        unsupported_reason = parsed.get("unsupported_reason")
        task_type = parsed.get("task_type")
        confidence = float(parsed.get("confidence", 0.9))

        raw_items = parsed.get("items", [])
        sub_items: List[ExtractedTaskSubItem] = []

        if isinstance(raw_items, list) and raw_items:
            for it in raw_items:
                if isinstance(it, dict) and it.get("prompt_tt") and it.get("expected_answer"):
                    sub_items.append(
                        ExtractedTaskSubItem(
                            task_type=it.get("task_type") or task_type or "custom",
                            prompt_tt=it.get("prompt_tt"),
                            expected_answer=it.get("expected_answer"),
                            cell_count=it.get("cell_count"),
                            grade_level=it.get("grade_level", grade_hint or 7),
                            topic_tag=it.get("topic_tag") or parsed.get("topic_tag"),
                            topic_name_tt=it.get("topic_name_tt") or parsed.get("topic_name_tt"),
                        )
                    )

        # Fallback to single top-level task if items list was empty
        prompt_tt = parsed.get("prompt_tt")
        expected_answer = parsed.get("expected_answer")
        cell_count = parsed.get("cell_count")
        grade_level = parsed.get("grade_level", grade_hint or 7)
        topic_tag = parsed.get("topic_tag")
        topic_name_tt = parsed.get("topic_name_tt")

        if not sub_items and prompt_tt and expected_answer:
            sub_items.append(
                ExtractedTaskSubItem(
                    task_type=task_type or "custom",
                    prompt_tt=prompt_tt,
                    expected_answer=expected_answer,
                    cell_count=cell_count,
                    grade_level=grade_level,
                    topic_tag=topic_tag,
                    topic_name_tt=topic_name_tt,
                )
            )

        if sub_items:
            is_supported = True
            if not prompt_tt:
                prompt_tt = sub_items[0].prompt_tt
            if not expected_answer:
                expected_answer = sub_items[0].expected_answer
            if not cell_count:
                cell_count = sub_items[0].cell_count
            if not task_type:
                task_type = sub_items[0].task_type

        return ExtractedTaskResult(
            is_supported=is_supported,
            unsupported_reason=unsupported_reason,
            task_type=task_type,
            prompt_tt=prompt_tt,
            expected_answer=expected_answer,
            cell_count=cell_count,
            grade_level=grade_level,
            topic_tag=topic_tag,
            topic_name_tt=topic_name_tt,
            items=sub_items,
            confidence=confidence,
            raw_ocr_text=ocr_text,
        )
