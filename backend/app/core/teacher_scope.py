from typing import Optional
from fastapi import HTTPException, status

from app.models.school import ClassModel


def normalize_teacher_uuid(raw: Optional[str], required: bool = False) -> Optional[str]:
    """Validate and trim an X-Teacher-UUID header value."""
    if raw is None:
        if required:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="X-Teacher-UUID header is required.",
            )
        return None

    cleaned = raw.strip()
    if not cleaned or len(cleaned) > 64:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="X-Teacher-UUID must be a non-empty string up to 64 characters.",
        )
    return cleaned


def assert_class_access(cls_obj: ClassModel, teacher_uuid: Optional[str]) -> None:
    """Enforce ownership isolation when both request header and class owner are set."""
    if teacher_uuid and cls_obj.teacher_uuid and cls_obj.teacher_uuid != teacher_uuid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to class '{cls_obj.class_id}'.",
        )
