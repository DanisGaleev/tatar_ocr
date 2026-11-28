"""Service layer for transient in-RAM blank photo recognition, verification, and grading."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.generators.checker import GradingScale, QuestionVerificationResult, verify_test_submission
from app.models.school import StudentModel
from app.models.submission import SubmissionModel
from app.models.teacher import TeacherModel
from app.models.test import AssembledTestModel
from app.schemas.submission import (
    QRMetadataItem,
    RecognizeBlankResponse,
    RecognizedBlankStats,
    RecognizedCellItem,
    RecognizedQuestionItem,
    SubmissionCellResult,
    SubmissionItem,
    SubmissionQuestionResult,
    TopPredictionItem,
)
from app.services.blank_scanner_bridge import (
    BlankScannerBridgeProtocol,
    RawBlankScanResult,
    RawScannedCell,
    get_blank_scanner_bridge,
)


class BlankImageValidationError(ValueError):
    """Raised when uploaded photo fails size/content validation."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class BlankAssignmentNotFoundError(LookupError):
    """Raised when an explicitly requested assignment or variant does not exist."""


class BlankAutoSaveError(ValueError):
    """Raised when auto_save=True cannot persist due to missing assignment or class context."""


class BlankRecognitionService:
    """Orchestrates transient blank photo recognition via C++ bridge and answer-key verification."""

    def __init__(self, bridge: Optional[BlankScannerBridgeProtocol] = None):
        self.bridge = bridge or get_blank_scanner_bridge()

    @staticmethod
    def validate_image_bytes(image_bytes: bytes, mime_type: Optional[str] = None) -> None:
        if mime_type is not None and not mime_type.lower().startswith("image/"):
            raise BlankImageValidationError(
                f"Invalid file content type '{mime_type}'. Must be an image (JPEG, PNG, WEBP).",
                status_code=400,
            )
        if not image_bytes:
            raise BlankImageValidationError("Uploaded image file is empty.", status_code=400)
        if len(image_bytes) > settings.BLANK_MAX_IMAGE_SIZE_BYTES:
            max_mb = settings.BLANK_MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
            raise BlankImageValidationError(
                f"Image size ({len(image_bytes)} bytes) exceeds {max_mb} MB limit.",
                status_code=413,
            )

    @staticmethod
    async def _resolve_teacher_preferences(
        db: AsyncSession,
        x_teacher_uuid: Optional[str],
        override_threshold_pct: Optional[float],
    ) -> tuple[GradingScale, float]:
        scale = GradingScale()
        threshold_pct = float(settings.OCR_DEFAULT_CONFIDENCE_THRESHOLD_PCT)

        if x_teacher_uuid:
            stmt = select(TeacherModel).where(TeacherModel.teacher_uuid == x_teacher_uuid)
            res = await db.execute(stmt)
            teacher = res.scalar_one_or_none()
            if teacher is not None:
                scale = GradingScale(
                    grade_5_min_pct=teacher.grade_5_min_pct,
                    grade_4_min_pct=teacher.grade_4_min_pct,
                    grade_3_min_pct=teacher.grade_3_min_pct,
                )
                if teacher.confidence_flag_threshold is not None:
                    raw_th = float(teacher.confidence_flag_threshold)
                    threshold_pct = raw_th * 100.0 if raw_th <= 1.0 else raw_th

        if override_threshold_pct is not None:
            threshold_pct = float(override_threshold_pct)

        return scale, round(threshold_pct, 1)

    @staticmethod
    async def _resolve_student(
        db: AsyncSession,
        student_id: Optional[str],
    ) -> Optional[StudentModel]:
        if not student_id:
            return None
        stmt = select(StudentModel).where(StudentModel.student_id == student_id)
        res = await db.execute(stmt)
        return res.scalar_one_or_none()

    @staticmethod
    async def _resolve_test_variant(
        db: AsyncSession,
        assignment_id: Optional[str],
        variant: int,
        explicit_assignment: bool,
        explicit_variant: bool,
    ) -> tuple[Optional[AssembledTestModel], Optional[dict[str, Any]], Optional[str]]:
        if not assignment_id:
            return None, None, None

        stmt = select(AssembledTestModel).where(AssembledTestModel.test_id == assignment_id)
        res = await db.execute(stmt)
        test_obj = res.scalar_one_or_none()

        if test_obj is None:
            msg = f"Assignment '{assignment_id}' not found."
            if explicit_assignment:
                raise BlankAssignmentNotFoundError(msg)
            return None, None, msg

        try:
            bundle = json.loads(test_obj.bundle_json)
        except (TypeError, ValueError):
            msg = f"Assignment '{assignment_id}' contains invalid bundle metadata."
            if explicit_assignment:
                raise BlankAssignmentNotFoundError(msg)
            return test_obj, None, msg

        variants = bundle.get("variants", [])
        matching_var = next((v for v in variants if v.get("variant_id") == variant), None)
        if matching_var is None:
            if explicit_variant or not variants:
                msg = f"Variant {variant} for assignment '{assignment_id}' not found."
                if explicit_assignment or explicit_variant:
                    raise BlankAssignmentNotFoundError(msg)
                return test_obj, None, msg
            matching_var = variants[0]

        return test_obj, matching_var, None

    @staticmethod
    def _build_cells(
        raw_cells: list[RawScannedCell],
        threshold_pct: float,
    ) -> tuple[list[RecognizedCellItem], int]:
        out: list[RecognizedCellItem] = []
        uncertain_count = 0
        for c in raw_cells:
            needs_review = (not c.is_empty) and (c.confidence < threshold_pct)
            if needs_review:
                uncertain_count += 1
            out.append(
                RecognizedCellItem(
                    cell_idx=c.cell_idx,
                    char=c.char,
                    confidence=c.confidence,
                    is_empty=c.is_empty,
                    review_required=needs_review,
                    top3=[TopPredictionItem(char=t.char, confidence=t.confidence) for t in c.top3],
                    bbox=list(c.bbox),
                )
            )
        return out, uncertain_count

    @staticmethod
    def _to_submission_question(qv: QuestionVerificationResult) -> SubmissionQuestionResult:
        return SubmissionQuestionResult(
            question_number=qv.question_number,
            marker_id=qv.marker_id,
            topic_tag=qv.topic_tag,
            is_correct=qv.is_correct,
            points_earned=qv.points_earned,
            cells=[
                SubmissionCellResult(
                    cell_index=cell.cell_index,
                    expected_char=cell.expected_char,
                    predicted_char=cell.predicted_char,
                    confidence=round(max(0.0, min(1.0, float(cell.confidence))), 4),
                    status=cell.status,
                    teacher_override=cell.teacher_override,
                )
                for cell in qv.cells
            ],
        )

    async def recognize_and_verify(
        self,
        db: AsyncSession,
        image_bytes: bytes,
        mime_type: Optional[str] = "image/jpeg",
        assignment_id: Optional[str] = None,
        variant: Optional[int] = None,
        student_id: Optional[str] = None,
        class_id: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        auto_save: bool = False,
        x_teacher_uuid: Optional[str] = None,
    ) -> RecognizeBlankResponse:
        self.validate_image_bytes(image_bytes, mime_type=mime_type)

        # C++20 OCR runs purely in RAM; native core releases the Python GIL.
        raw_scan: RawBlankScanResult = await run_in_threadpool(self.bridge.scan_image_bytes, image_bytes)

        qr_meta = (
            QRMetadataItem(
                raw=raw_scan.qr_payload.raw,
                assignment_id=raw_scan.qr_payload.assignment_id,
                variant=raw_scan.qr_payload.variant,
                page=raw_scan.qr_payload.page,
                total_pages=raw_scan.qr_payload.total_pages,
                questions_count=raw_scan.qr_payload.questions_count,
                student_id=raw_scan.qr_payload.student_id,
            )
            if raw_scan.qr_payload is not None
            else None
        )

        explicit_assignment = bool(assignment_id and assignment_id.strip())
        explicit_variant = variant is not None

        resolved_assignment_id = (
            assignment_id.strip()
            if explicit_assignment
            else (qr_meta.assignment_id if qr_meta and qr_meta.assignment_id else None)
        )
        resolved_variant = (
            variant
            if variant is not None
            else (qr_meta.variant if qr_meta and qr_meta.variant is not None else 1)
        )
        resolved_student_id = (
            student_id.strip()
            if student_id and student_id.strip()
            else (qr_meta.student_id if qr_meta and qr_meta.student_id else None)
        )

        grading_scale, threshold_pct = await self._resolve_teacher_preferences(
            db=db,
            x_teacher_uuid=x_teacher_uuid,
            override_threshold_pct=confidence_threshold,
        )

        student_obj = await self._resolve_student(db, resolved_student_id)
        resolved_class_id = class_id.strip() if class_id and class_id.strip() else (student_obj.class_id if student_obj else None)
        resolved_student_name = raw_scan.student_name or (student_obj.full_name if student_obj else "")

        name_cells, _ = self._build_cells(raw_scan.student_name_cells, threshold_pct)

        recognized_questions_raw: list[RecognizedQuestionItem] = []
        uncertain_total = 0
        ocr_submissions_payload: list[dict[str, Any]] = []

        for q in raw_scan.questions:
            q_cells, q_uncertain = self._build_cells(q.cells, threshold_pct)
            uncertain_total += q_uncertain
            recognized_questions_raw.append(
                RecognizedQuestionItem(
                    q_num=q.q_num,
                    marker_id=10 + q.q_num,
                    text=q.text,
                    cells=q_cells,
                )
            )
            # C++ NativeBlankScanner emits 1-based cell_idx; checker.py expects 0-based cell_index.
            ocr_submissions_payload.append(
                {
                    "question_number": q.q_num,
                    "cells": [
                        {
                            "cell_index": max(0, c.cell_idx - 1),
                            "predicted_char": c.char,
                            "confidence": round(c.confidence / 100.0, 4),
                        }
                        for c in q.cells
                    ],
                }
            )

        test_obj, variant_obj, verification_warning = await self._resolve_test_variant(
            db=db,
            assignment_id=resolved_assignment_id,
            variant=resolved_variant,
            explicit_assignment=explicit_assignment,
            explicit_variant=explicit_variant,
        )

        is_verified = False
        overall_score: Optional[float] = None
        max_score: Optional[float] = None
        score_percentage: Optional[float] = None
        final_grade: Optional[int] = None
        questions_results: list[SubmissionQuestionResult] = []
        submission_draft: Optional[SubmissionItem] = None
        saved_submission_id: Optional[str] = None

        if test_obj is not None and variant_obj is not None:
            bundle_questions = variant_obj.get("questions", [])
            if bundle_questions:
                verification = verify_test_submission(
                    questions=bundle_questions,
                    submissions=ocr_submissions_payload,
                    grading_scale=grading_scale,
                )
                is_verified = True
                overall_score = verification.overall_score
                max_score = verification.max_score
                score_percentage = verification.score_percentage
                final_grade = verification.final_grade

                meta_by_q = {q["question_number"]: q for q in bundle_questions}
                ver_by_q = {q.question_number: q for q in verification.questions_results}

                for rq in recognized_questions_raw:
                    q_meta = meta_by_q.get(rq.q_num)
                    q_ver = ver_by_q.get(rq.q_num)
                    if q_meta is not None:
                        rq.marker_id = int(q_meta.get("marker_id", rq.marker_id))
                        rq.prompt = q_meta.get("prompt")
                        rq.topic_tag = q_meta.get("topic_tag")
                        rq.expected_answer = q_meta.get("expected_answer")
                    if q_ver is not None:
                        rq.is_correct = q_ver.is_correct
                        rq.points_earned = q_ver.points_earned

                questions_results = [
                    self._to_submission_question(qv) for qv in verification.questions_results
                ]

                submission_draft = SubmissionItem(
                    client_submission_uuid=str(uuid.uuid4()),
                    student_id=resolved_student_id or "unknown",
                    student_name=resolved_student_name or "Неизвестный ученик",
                    variant=int(variant_obj.get("variant_id", resolved_variant)),
                    checked_at=datetime.now(timezone.utc),
                    overall_score=overall_score,
                    max_score=max_score,
                    final_grade=final_grade,
                    teacher_reviewed_flags=uncertain_total,
                    questions_results=questions_results,
                )
            else:
                verification_warning = f"Variant {resolved_variant} of assignment '{resolved_assignment_id}' has no questions."

        if auto_save:
            if not is_verified or submission_draft is None or not resolved_assignment_id:
                raise BlankAutoSaveError(
                    "Cannot auto-save submission: no matching assignment/variant could be verified."
                )
            if not resolved_class_id:
                raise BlankAutoSaveError(
                    "Cannot auto-save submission: class_id is required when student_id is not linked to a class."
                )
            saved_submission_id = f"sub_{uuid.uuid4().hex[:8]}"
            sub_model = SubmissionModel(
                submission_id=saved_submission_id,
                client_submission_uuid=submission_draft.client_submission_uuid,
                assignment_id=resolved_assignment_id,
                class_id=resolved_class_id,
                student_id=submission_draft.student_id,
                student_name=submission_draft.student_name,
                variant=submission_draft.variant,
                checked_at=submission_draft.checked_at or datetime.now(timezone.utc),
                overall_score=submission_draft.overall_score,
                max_score=submission_draft.max_score,
                final_grade=submission_draft.final_grade,
                teacher_reviewed_flags=submission_draft.teacher_reviewed_flags,
                questions_results_json=json.dumps(
                    [q.model_dump() for q in submission_draft.questions_results],
                    ensure_ascii=False,
                ),
            )
            db.add(sub_model)
            await db.commit()

        zero_touch = (uncertain_total == 0) and (raw_scan.stats.total_letters > 0)
        stats = RecognizedBlankStats(
            total_letters=raw_scan.stats.total_letters,
            avg_confidence=raw_scan.stats.avg_confidence,
            questions_count=raw_scan.stats.questions_count,
            uncertain_cells_count=uncertain_total,
            zero_touch=zero_touch,
        )

        return RecognizeBlankResponse(
            status=raw_scan.status,
            rectification_method=raw_scan.rectification_method,
            qr_metadata=qr_meta,
            assignment_id=resolved_assignment_id,
            variant=int(variant_obj.get("variant_id", resolved_variant)) if variant_obj else resolved_variant,
            student_id=resolved_student_id,
            class_id=resolved_class_id,
            student_name=resolved_student_name,
            student_name_cells=name_cells,
            questions=recognized_questions_raw,
            stats=stats,
            confidence_threshold_pct=threshold_pct,
            is_verified=is_verified,
            verification_warning=verification_warning,
            overall_score=overall_score,
            max_score=max_score,
            score_percentage=score_percentage,
            final_grade=final_grade,
            teacher_reviewed_flags=uncertain_total,
            questions_results=questions_results,
            submission_draft=submission_draft,
            saved_submission_id=saved_submission_id,
        )


def get_blank_recognition_service() -> BlankRecognitionService:
    return BlankRecognitionService()
