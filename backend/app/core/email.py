from datetime import datetime, timezone
from email.message import EmailMessage
import logging
from typing import Any, Dict, List, Optional

import aiosmtplib

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    def __init__(self) -> None:
        self.mock_mode: bool = settings.EMAIL_MOCK_MODE
        # In-memory store for unit and integration testing
        self.sent_codes: List[Dict[str, Any]] = []

    def clear_sent_codes(self) -> None:
        """Clear recorded mock codes."""
        self.sent_codes.clear()

    def get_last_code_for(self, email: str, code_type: Optional[str] = None) -> Optional[str]:
        """Retrieve most recent verification code sent to an email."""
        for item in reversed(self.sent_codes):
            if item["to_email"].lower() == email.lower():
                if code_type is None or item["code_type"] == code_type:
                    return item["code"]
        return None

    async def send_verification_code(
        self,
        to_email: str,
        code: str,
        code_type: str = "register_verify",
    ) -> bool:
        if self.mock_mode:
            record = {
                "to_email": to_email,
                "code": code,
                "code_type": code_type,
                "timestamp": datetime.now(timezone.utc),
            }
            self.sent_codes.append(record)
            logger.info(
                f"[EMAIL MOCK] Code sent to {to_email} ({code_type}): {code}"
            )
            return True

        # Real SMTP transport
        subject_map = {
            "register_verify": "Код подтверждения регистрации — «Кара куян» (Tatar OCR)",
            "login_2fa": "Код двухфакторной аутентификации — «Кара куян» (Tatar OCR)",
            "password_reset": "Код для сброса пароля — «Кара куян» (Tatar OCR)",
        }
        subject = subject_map.get(code_type, "Код безопасности — «Кара куян» (Tatar OCR)")

        body = (
            f"Здравствуйте!\n\n"
            f"Ваш проверочный код: {code}\n\n"
            f"Код действителен в течение 10 минут.\n"
            f"Если вы не совершали этот запрос, проигнорируйте данное письмо.\n\n"
            f"— Команда проекта «Кара куян»"
        )

        try:
            message = EmailMessage()
            message["From"] = settings.EMAILS_FROM_EMAIL
            message["To"] = to_email
            message["Subject"] = subject
            message.set_content(body)

            await aiosmtplib.send(
                message,
                hostname=settings.SMTP_HOST,
                port=settings.SMTP_PORT,
                username=settings.SMTP_USER or None,
                password=settings.SMTP_PASSWORD or None,
                start_tls=settings.SMTP_TLS,
            )
            return True
        except Exception as exc:
            logger.error(f"Failed to send email to {to_email}: {exc}")
            return False

    async def send_password_reset_code(self, to_email: str, code: str) -> bool:
        return await self.send_verification_code(to_email, code, code_type="password_reset")

    async def send_2fa_code(self, to_email: str, code: str) -> bool:
        return await self.send_verification_code(to_email, code, code_type="login_2fa")


email_service = EmailService()
