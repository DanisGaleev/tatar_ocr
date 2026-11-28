"""Bridge layer between the C++20 native OCR core (tatar_ocr_native) and backend services."""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import cv2
import numpy as np

from app.core.config import settings

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOAD_LOCK = threading.Lock()
_DLL_DIR_HANDLES: list[Any] = []
_REGISTERED_DLL_DIRS: set[Path] = set()
_NATIVE_MODULE: Any = None


class InvalidBlankImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as a valid non-empty image."""


class NativeScannerUnavailableError(RuntimeError):
    """Raised when tatar_ocr_native C++ module, spec, or ONNX weights cannot be loaded."""


@dataclass(frozen=True)
class RawTopPrediction:
    char: str
    confidence: float


@dataclass(frozen=True)
class RawScannedCell:
    cell_idx: int
    char: str
    confidence: float
    is_empty: bool
    top3: list[RawTopPrediction] = field(default_factory=list)
    bbox: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class RawQuestionResult:
    q_num: int
    text: str
    cells: list[RawScannedCell] = field(default_factory=list)


@dataclass(frozen=True)
class RawBlankStats:
    total_letters: int
    avg_confidence: float
    questions_count: int


@dataclass(frozen=True)
class RawQRPayload:
    raw: str
    assignment_id: Optional[str] = None
    variant: Optional[int] = None
    page: Optional[int] = None
    total_pages: Optional[int] = None
    questions_count: Optional[int] = None
    student_id: Optional[str] = None


@dataclass(frozen=True)
class RawBlankScanResult:
    status: str
    rectification_method: str
    student_name: str
    student_name_cells: list[RawScannedCell]
    questions: list[RawQuestionResult]
    stats: RawBlankStats
    qr_payload: Optional[RawQRPayload] = None


class BlankScannerBridgeProtocol(Protocol):
    """Abstraction over the C++20 blank OCR pipeline for service decoupling and testing."""

    def scan_image_bytes(self, image_bytes: bytes) -> RawBlankScanResult:
        """Decode image bytes in RAM, run OCR + QR detection, and return structured results."""
        ...


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    candidate = _REPO_ROOT / path
    return candidate


def _load_native_module(native_build_dir: Optional[str | Path] = None) -> Any:
    global _NATIVE_MODULE
    with _LOAD_LOCK:
        if _NATIVE_MODULE is not None:
            return _NATIVE_MODULE

        candidates: list[Path] = []
        env_native_dir = native_build_dir or settings.NATIVE_BUILD_DIR or os.environ.get("TATAR_OCR_NATIVE_DIR")
        if env_native_dir:
            env_path = _resolve_repo_path(env_native_dir)
            candidates.extend([env_path, env_path / "Release"])

        candidates.extend([
            _REPO_ROOT / "native" / "build" / "Release",
            _REPO_ROOT / "native" / "build",
            _REPO_ROOT,
        ])

        for path in candidates:
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))

        if hasattr(os, "add_dll_directory"):
            dll_dirs = [p for p in candidates if p.is_dir()]
            native_deps = _REPO_ROOT / ".native_deps"
            if native_deps.is_dir():
                for dll_file in native_deps.rglob("*.dll"):
                    if dll_file.parent not in dll_dirs:
                        dll_dirs.append(dll_file.parent)
            for dll_dir in dll_dirs:
                if dll_dir not in _REGISTERED_DLL_DIRS:
                    _REGISTERED_DLL_DIRS.add(dll_dir)
                    _DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))

        try:
            import tatar_ocr_native
        except ImportError as exc:
            raise NativeScannerUnavailableError(
                "C++20 native OCR module 'tatar_ocr_native' is not available. "
                "Build it with: cmake -S native -B native/build && cmake --build native/build --config Release "
                "or set NATIVE_BUILD_DIR / TATAR_OCR_NATIVE_DIR."
            ) from exc

        _NATIVE_MODULE = tatar_ocr_native
        return _NATIVE_MODULE


def _safe_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_qr_string(raw_qr: str) -> Optional[RawQRPayload]:
    cleaned = (raw_qr or "").strip()
    if not cleaned:
        return None

    assignment_id: Optional[str] = None
    variant: Optional[int] = None
    page: Optional[int] = None
    total_pages: Optional[int] = None
    questions_count: Optional[int] = None
    student_id: Optional[str] = None

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            tid = parsed.get("tid") or parsed.get("assignment_id")
            if tid is not None:
                assignment_id = str(tid).strip() or None
            variant = _safe_int(parsed.get("var") if "var" in parsed else parsed.get("variant"))
            page = _safe_int(parsed.get("page"))
            total_pages = _safe_int(parsed.get("tot") if "tot" in parsed else parsed.get("total_pages"))
            questions_count = _safe_int(parsed.get("n_q") if "n_q" in parsed else parsed.get("questions_count"))
            stu = parsed.get("stu_id") or parsed.get("student_id")
            if stu is not None:
                student_id = str(stu).strip() or None
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return RawQRPayload(
        raw=cleaned,
        assignment_id=assignment_id,
        variant=variant,
        page=page,
        total_pages=total_pages,
        questions_count=questions_count,
        student_id=student_id,
    )


def _detect_qr_payload(raw_bgr: np.ndarray, rectified_bgr: Optional[np.ndarray]) -> Optional[RawQRPayload]:
    detector = cv2.QRCodeDetector()
    candidates: list[np.ndarray] = []

    if rectified_bgr is not None and getattr(rectified_bgr, "size", 0) > 0:
        h, w = rectified_bgr.shape[:2]
        # Canonical A4 header QR zone (x~230..410, y~60..240 on 2100x2970) with generous margin
        y1, y2 = 0, min(h, max(1, int(h * 0.16)))
        x1, x2 = max(0, int(w * 0.05)), min(w, max(1, int(w * 0.35)))
        if y2 > y1 and x2 > x1:
            candidates.append(rectified_bgr[y1:y2, x1:x2])
        candidates.append(rectified_bgr)

    candidates.append(raw_bgr)

    for img in candidates:
        try:
            decoded, _, _ = detector.detectAndDecode(img)
        except cv2.error:
            continue
        if decoded and decoded.strip():
            return _parse_qr_string(decoded)
    return None


def _convert_cell(cell_dict: dict[str, Any]) -> RawScannedCell:
    top3_raw = cell_dict.get("top3") or []
    top3 = [
        RawTopPrediction(char=str(ch), confidence=round(float(conf), 1))
        for ch, conf in top3_raw[:3]
    ]
    bbox_raw = cell_dict.get("bbox") or []
    bbox = [int(v) for v in bbox_raw[:4]]
    return RawScannedCell(
        cell_idx=int(cell_dict.get("cell_idx", 0)),
        char=str(cell_dict.get("char", " ")),
        confidence=round(float(cell_dict.get("confidence", 0.0)), 1),
        is_empty=bool(cell_dict.get("is_empty", True)),
        top3=top3,
        bbox=bbox,
    )


class NativeBlankScannerBridge:
    """Thread-safe in-memory bridge to tatar_ocr::NativeBlankScanner (C++20 + ONNX Runtime)."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        spec_path: Optional[str | Path] = None,
        native_build_dir: Optional[str | Path] = None,
    ):
        self.model_path = _resolve_repo_path(model_path or settings.OCR_MODEL_PATH)
        self.spec_path = _resolve_repo_path(spec_path or settings.OCR_SPEC_PATH)
        self.native_build_dir = native_build_dir
        self._scanner_lock = threading.Lock()
        self._scanner: Any = None

    def _get_scanner(self) -> Any:
        if self._scanner is not None:
            return self._scanner
        with self._scanner_lock:
            if self._scanner is not None:
                return self._scanner
            if not self.model_path.exists():
                raise NativeScannerUnavailableError(f"ONNX model weights not found: {self.model_path}")
            if not self.spec_path.exists():
                raise NativeScannerUnavailableError(f"Pipeline spec JSON not found: {self.spec_path}")
            native_mod = _load_native_module(self.native_build_dir)
            self._scanner = native_mod.NativeBlankScanner(str(self.model_path), str(self.spec_path))
            return self._scanner

    def scan_image_bytes(self, image_bytes: bytes) -> RawBlankScanResult:
        if not image_bytes:
            raise InvalidBlankImageError("Uploaded image file is empty.")

        np_buf = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        if img_bgr is None or img_bgr.size == 0:
            raise InvalidBlankImageError("Could not decode image bytes. Provide a valid JPEG, PNG, or WEBP photo.")

        scanner = self._get_scanner()
        contiguous_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
        native_res = scanner.process_blank(contiguous_bgr)

        rectified_bgr = native_res.get("rectified_bgr")
        qr_payload = _detect_qr_payload(contiguous_bgr, rectified_bgr)

        name_cells = [_convert_cell(c) for c in native_res.get("student_name_cells", [])]
        questions = [
            RawQuestionResult(
                q_num=int(q.get("q_num", idx)),
                text=str(q.get("text", "")),
                cells=[_convert_cell(c) for c in q.get("cells", [])],
            )
            for idx, q in enumerate(native_res.get("questions", []), start=1)
        ]
        stats_raw = native_res.get("stats") or {}
        stats = RawBlankStats(
            total_letters=int(stats_raw.get("total_letters", 0)),
            avg_confidence=round(float(stats_raw.get("avg_confidence", 0.0)), 1),
            questions_count=int(stats_raw.get("questions_count", len(questions))),
        )

        return RawBlankScanResult(
            status=str(native_res.get("status", "success")),
            rectification_method=str(native_res.get("rectification_method", "unknown")),
            student_name=str(native_res.get("student_name", "")),
            student_name_cells=name_cells,
            questions=questions,
            stats=stats,
            qr_payload=qr_payload,
        )


_DEFAULT_BRIDGE_LOCK = threading.Lock()
_DEFAULT_BRIDGE: Optional[NativeBlankScannerBridge] = None


def get_blank_scanner_bridge() -> NativeBlankScannerBridge:
    """Return a singleton NativeBlankScannerBridge so the ONNX session is loaded once."""
    global _DEFAULT_BRIDGE
    if _DEFAULT_BRIDGE is not None:
        return _DEFAULT_BRIDGE
    with _DEFAULT_BRIDGE_LOCK:
        if _DEFAULT_BRIDGE is None:
            _DEFAULT_BRIDGE = NativeBlankScannerBridge()
        return _DEFAULT_BRIDGE
