import base64
import logging
import httpx
from typing import Optional
from app.core.config import settings
from app.services.task_extractor.base import BaseOCRClient

logger = logging.getLogger(__name__)

class YandexOCRClient(BaseOCRClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        folder_id: Optional[str] = None,
        ocr_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or settings.YANDEX_API_KEY
        self.folder_id = folder_id or settings.YANDEX_FOLDER_ID
        self.ocr_url = ocr_url or settings.YANDEX_OCR_URL
        self.timeout = timeout

    def _normalize_mime_type(self, mime_type: str) -> str:
        mime = mime_type.lower().strip()
        if "png" in mime:
            return "PNG"
        elif "webp" in mime:
            return "WEBP"
        elif "pdf" in mime:
            return "PDF"
        return "JPEG"

    async def recognize_text(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        if not self.api_key:
            raise ValueError("Yandex API key is not configured for OCR service.")

        encoded_content = base64.b64encode(image_bytes).decode("utf-8")
        normalized_mime = self._normalize_mime_type(mime_type)

        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.folder_id:
            headers["x-folder-id"] = self.folder_id

        payload = {
            "mimeType": normalized_mime,
            "languageCodes": ["tt", "ru", "en"],
            "model": "page",
            "content": encoded_content,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.ocr_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                err_detail = exc.response.text or str(exc)
                logger.error("Yandex OCR HTTP error %s: %s", exc.response.status_code, err_detail)
                raise RuntimeError(f"Yandex OCR HTTP {exc.response.status_code}: {err_detail}") from exc
            except httpx.TimeoutException as exc:
                logger.error("Yandex OCR request timed out after %s seconds", self.timeout)
                raise RuntimeError(f"Yandex OCR timeout: запрос превысил лимит ожидания ({self.timeout}с). Попробуйте еще раз.") from exc
            except Exception as exc:
                err_msg = str(exc) or repr(exc) or type(exc).__name__
                logger.error("Yandex OCR connection error: %s (%s)", type(exc).__name__, err_msg)
                raise RuntimeError(f"Yandex OCR {type(exc).__name__}: {err_msg}") from exc

        # Extract recognized text
        text_annotation = data.get("result", {}).get("textAnnotation", {})
        full_text = text_annotation.get("fullText", "").strip()
        if full_text:
            return full_text

        # Fallback to reconstructing lines from blocks
        lines = []
        for block in text_annotation.get("blocks", []):
            for line in block.get("lines", []):
                text = line.get("text", "").strip()
                if text:
                    lines.append(text)

        return "\n".join(lines).strip()
