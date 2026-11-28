import io
import os
import json
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import qrcode

_MPL_DIR = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib
matplotlib.use('Agg')
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as patches
import matplotlib.font_manager as fm

# Setup font for Tatar Cyrillic characters (cross-platform Linux & Windows resolution)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FONT_CANDIDATES = [
    _PROJECT_ROOT / "fonts" / "Rubik-Regular.ttf",
    _PROJECT_ROOT / "fonts" / "PlaypenSans.ttf",
    Path("/app/fonts/Rubik-Regular.ttf"),
    Path("/app/fonts/PlaypenSans.ttf"),
    Path("fonts/Rubik-Regular.ttf"),
    Path("fonts/PlaypenSans.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
]

_FONT_PATH = next((p for p in _FONT_CANDIDATES if p.exists()), None)
_FONT_NAME = "TatarFont"

if _FONT_PATH is not None and _FONT_PATH.exists():
    _fe = fm.FontEntry(fname=str(_FONT_PATH.resolve()), name=_FONT_NAME)
    fm.fontManager.ttflist.append(_fe)
else:
    _FONT_NAME = "sans-serif"

_ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
}
_ARUCO_DICT_CACHE: Dict[int, Any] = {}


@lru_cache(maxsize=1)
def _load_pipeline_spec() -> Dict[str, Any]:
    spec_path = _PROJECT_ROOT / "pipeline_spec.json"
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_aruco_img(marker_id: int, size_px: int = 140, dict_type=cv2.aruco.DICT_4X4_50) -> np.ndarray:
    aruco_dict = _ARUCO_DICT_CACHE.get(dict_type)
    if aruco_dict is None:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        _ARUCO_DICT_CACHE[dict_type] = aruco_dict
    return cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)


def generate_qr_img(payload_str: str, box_size: int = 10) -> np.ndarray:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=2,
    )
    qr.add_data(payload_str)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return np.array(img.convert('RGB'))


def _draw_blank_page(
    ax: Axes,
    assignment_id: str,
    title: str,
    variant_id: int,
    questions: List[Dict[str, Any]],
    student_name: Optional[str] = None,
    student_id: Optional[str] = None,
    language: str = "tt",
):
    """
    Renders elements of an official A4 (210 x 297 mm) OCR test blank onto given Axes
    strictly aligned with pipeline_spec.json SSOT coordinates (10 px/mm).
    """
    spec = _load_pipeline_spec()
    canvas_spec = spec["canvas"]
    aruco_spec = spec["aruco"]
    sn_spec = spec["grid"]["student_name"]
    q_spec = spec["grid"]["questions"]

    px_per_mm = 10.0
    page_w_mm = canvas_spec["width"] / px_per_mm
    page_h_mm = canvas_spec["height"] / px_per_mm
    dict_type = _ARUCO_DICT_MAP.get(aruco_spec.get("dict", "DICT_4X4_50"), cv2.aruco.DICT_4X4_50)

    # 1. Page Corner ArUco Markers aligned with canvas.dst_corners (6mm..204mm, 6mm..291mm)
    corner_size_mm = 14.0
    margin_corner_mm = canvas_spec["dst_corners"][0][0] / px_per_mm

    corner_positions = {
        0: (margin_corner_mm, margin_corner_mm),
        1: (page_w_mm - margin_corner_mm - corner_size_mm, margin_corner_mm),
        2: (margin_corner_mm, page_h_mm - margin_corner_mm - corner_size_mm),
        3: (page_w_mm - margin_corner_mm - corner_size_mm, page_h_mm - margin_corner_mm - corner_size_mm),
    }

    for cid, (cx, cy) in corner_positions.items():
        c_img = generate_aruco_img(cid, size_px=140, dict_type=dict_type)
        ax.imshow(c_img, cmap='gray', extent=(cx, cx + corner_size_mm, cy + corner_size_mm, cy), zorder=10)

    max_questions = int(q_spec["max_questions"])
    rendered_questions = questions[:max_questions]

    # 2. Header Section
    header_top_y = 6.0
    header_left_x = 23.0

    # QR Code with Test Metadata (includes stu_id when personalized, lang when non-tt)
    qr_dict: Dict[str, Any] = {
        "tid": assignment_id,
        "var": variant_id,
        "page": 1,
        "tot": 1,
        "n_q": len(rendered_questions),
    }
    if student_id:
        qr_dict["stu_id"] = student_id
    if language != "tt":
        qr_dict["lang"] = language
    qr_payload = json.dumps(qr_dict, separators=(',', ':'), ensure_ascii=False)

    qr_img = generate_qr_img(qr_payload)
    qr_size_mm = 18.0
    qr_x = header_left_x
    qr_y = header_top_y
    ax.imshow(qr_img, extent=(qr_x, qr_x + qr_size_mm, qr_y + qr_size_mm, qr_y), zorder=10)

    ax.text(
        qr_x + qr_size_mm / 2.0, qr_y + qr_size_mm + 2.2,
        f"{assignment_id} (Вар. {variant_id})",
        fontsize=5.5, fontname=_FONT_NAME, ha='center', va='top', color='#475569'
    )

    # Header Title, Warnings & Student Identification
    name_cells_start_x = sn_spec["default_start_x"] / px_per_mm
    title_x = name_cells_start_x
    title_str = (title or "").strip()
    if title_str:
        title_display = title_str if len(title_str) <= 68 else title_str[:65] + "..."
        ax.text(title_x, header_top_y + 0.5, title_display,
                fontsize=7.6, fontname=_FONT_NAME, weight='bold', ha='left', va='top', color='#0f172a')

    ax.text(title_x, header_top_y + 5.2, "Внимание: заполняйте печатными заглавными буквами",
            fontsize=6.9, fontname=_FONT_NAME, ha='left', va='top', color='#334155')

    # Student Identification Field (15 cells x 8.8 mm = 132 mm, matching pipeline_spec.json)
    ax.text(title_x, header_top_y + 10.6, "ФИО ученика:",
            fontsize=7.2, fontname=_FONT_NAME, ha='left', va='top', color='#1e293b')

    name_cell_w = sn_spec["default_pitch"] / px_per_mm
    name_cells_start_y = sn_spec["default_top_y"] / px_per_mm
    name_cell_h = (sn_spec["default_bot_y"] - sn_spec["default_top_y"]) / px_per_mm
    num_name_cells = int(sn_spec["cell_count"])

    name_letters = list(student_name.upper()) if student_name else []

    for i in range(num_name_cells):
        nx = name_cells_start_x + i * name_cell_w
        rect = patches.Rectangle((nx, name_cells_start_y), name_cell_w, name_cell_h,
                                 linewidth=0.8, edgecolor='#000000', facecolor='none')
        ax.add_patch(rect)
        if i < len(name_letters) and name_letters[i] != ' ':
            ax.text(nx + name_cell_w / 2.0, name_cells_start_y + name_cell_h / 2.0,
                    name_letters[i], fontsize=7.5, fontname=_FONT_NAME,
                    ha='center', va='center', color='#0f172a', weight='bold')

    # Header separator
    header_sep_y = 34.5
    ax.plot([margin_corner_mm, page_w_mm - margin_corner_mm], [header_sep_y, header_sep_y],
            color='#cbd5e1', linewidth=0.8)

    # 3. Modular Question Blocks (aligned with canonical_anchors & grid.questions)
    q_marker_size_mm = 11.0
    base_id = int(aruco_spec.get("question_marker_base_id", 10))
    anchors = aruco_spec.get("canonical_anchors", {})
    cells_x = q_spec["default_start_x"] / px_per_mm
    cell_w_mm = q_spec["default_pitch"] / px_per_mm
    cell_h_mm = q_spec["default_row_h"] / px_per_mm
    default_top_offset_mm = q_spec["default_top_offset"] / px_per_mm
    fallback_base_y_px = q_spec["fallback_base_y"]
    fallback_step_y_px = q_spec["fallback_step_y"]

    for idx, q in enumerate(rendered_questions):
        marker_id = int(q.get("marker_id", base_id + idx + 1))
        anchor_px = anchors.get(str(marker_id), [125.0, fallback_base_y_px + idx * fallback_step_y_px])
        cx_mm = anchor_px[0] / px_per_mm
        cy_mm = anchor_px[1] / px_per_mm

        marker_x = cx_mm - q_marker_size_mm / 2.0
        marker_y = cy_mm - q_marker_size_mm / 2.0
        m_img = generate_aruco_img(marker_id, size_px=110, dict_type=dict_type)
        ax.imshow(m_img, cmap='gray',
                  extent=(marker_x, marker_x + q_marker_size_mm, marker_y + q_marker_size_mm, marker_y),
                  zorder=10)

        ax.text(marker_x + q_marker_size_mm / 2.0, marker_y + q_marker_size_mm + 1.5,
                f"ID {marker_id}", fontsize=5.5, fontname=_FONT_NAME, ha='center', va='top', color='#64748b')

        cells_y = cy_mm + default_top_offset_mm

        # Question Prompt Text & Dynamic Full-Line Wrapping (never truncate lines)
        default_q_label = f"Вопрос №{idx+1}" if language == "ru" else f"Сорау №{idx+1}"
        prompt_text = q.get("prompt", q.get("prompt_tt", default_q_label))
        q_prefix = f"{idx+1}) "

        def _wrap_lines(raw_text: str, width: int) -> List[str]:
            wrapped: List[str] = []
            for rl in raw_text.split("\n"):
                if len(rl) > width:
                    wrapped.extend(textwrap.wrap(rl, width=width))
                else:
                    wrapped.append(rl)
            return wrapped

        lines = _wrap_lines(prompt_text, 75)
        if len(lines) >= 3:
            lines = _wrap_lines(prompt_text, 96)

        if len(lines) >= 4:
            font_size = 5.4
            text_y = cells_y - 12.2
        elif len(lines) == 3:
            font_size = 6.2
            text_y = cells_y - 11.2
        elif len(lines) == 2:
            font_size = 7.2
            text_y = cells_y - 9.5
        else:
            font_size = 8.0 if len(q_prefix + lines[0]) <= 65 else 7.0
            text_y = cells_y - 7.5

        if len(lines) > 1:
            q_full_text = q_prefix + lines[0] + "\n   " + "\n   ".join(lines[1:])
        else:
            q_full_text = q_prefix + lines[0]

        ax.text(cells_x, text_y, q_full_text, fontsize=font_size, fontname=_FONT_NAME,
                ha='left', va='top', color='#0f172a')

        # 10mm x 10mm Answer Cells (1..12)
        cell_count = max(1, min(int(q.get("cell_count", 8)), 12))

        for c in range(cell_count):
            cx = cells_x + c * cell_w_mm
            rect = patches.Rectangle((cx, cells_y), cell_w_mm, cell_h_mm,
                                     linewidth=0.8, edgecolor='#000000', facecolor='none')
            ax.add_patch(rect)

        if cell_count == 1:
            single_hint = "<- 1 буква" if language == "ru" else "<- 1 хәреф"
            ax.text(cells_x + cell_w_mm + 2.5, cells_y + cell_h_mm / 2.0, single_hint,
                    fontsize=6.5, fontname=_FONT_NAME, ha='left', va='center', color='#64748b')

        # Separator line between questions
        if idx < len(rendered_questions) - 1:
            sep_y = cells_y + cell_h_mm + 3.2
            ax.plot([20.0, page_w_mm - 10.0], [sep_y, sep_y], color='#e2e8f0', linewidth=0.6, linestyle='--')

    # 4. Footer Instructions
    footer_y = 267.0
    ax.plot([margin_corner_mm, page_w_mm - margin_corner_mm], [footer_y, footer_y], color='#cbd5e1', linewidth=0.8)

    if language == "ru":
        inst_text = (
            "Внимание: бланк не сгибать. Пишите печатными заглавными буквами строго внутри клеток.\n"
            "Русский алфавит: А–Я, Ё. Не закрашивать маркеры ArUco и QR-код."
        )
    else:
        inst_text = (
            "Внимание: бланк не сгибать. Пишите печатными заглавными буквами строго внутри клеток.\n"
            "Татарские буквы: Ә, Җ, Ң, Ө, Ү, Һ. Не закрашивать маркеры ArUco и QR-код."
        )
    ax.text(page_w_mm / 2.0, footer_y + 2.5, inst_text,
            fontsize=7.0, fontname=_FONT_NAME, ha='center', va='top', color='#475569', multialignment='center')


def _create_blank_figure(
    assignment_id: str,
    title: str,
    variant_id: int,
    questions: List[Dict[str, Any]],
    student_name: Optional[str] = None,
    student_id: Optional[str] = None,
    language: str = "tt",
    dpi: int = 300,
) -> Figure:
    spec = _load_pipeline_spec()
    page_w_mm = spec["canvas"]["width"] / 10.0
    page_h_mm = spec["canvas"]["height"] / 10.0
    fig_w_in, fig_h_in = page_w_mm / 25.4, page_h_mm / 25.4

    fig = Figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w_mm)
    ax.set_ylim(0, page_h_mm)
    ax.invert_yaxis()
    ax.axis('off')

    _draw_blank_page(
        ax,
        assignment_id=assignment_id,
        title=title,
        variant_id=variant_id,
        questions=questions,
        student_name=student_name,
        student_id=student_id,
        language=language,
    )
    return fig


def render_blank_image(
    assignment_id: str,
    title: str,
    variant_id: int,
    questions: List[Dict[str, Any]],
    student_name: Optional[str] = None,
    student_id: Optional[str] = None,
    language: str = "tt",
) -> np.ndarray:
    """
    Renders the blank onto a canonical 2100x2970 BGR canvas image (for OCR E2E verification).
    """
    fig = _create_blank_figure(
        assignment_id=assignment_id,
        title=title,
        variant_id=variant_id,
        questions=questions,
        student_name=student_name,
        student_id=student_id,
        language=language,
        dpi=254,
    )
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    spec = _load_pipeline_spec()
    target_w = int(spec["canvas"]["width"])
    target_h = int(spec["canvas"]["height"])
    if bgr.shape[1] != target_w or bgr.shape[0] != target_h:
        bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return bgr


def render_blank_pdf(
    assignment_id: str,
    title: str,
    variant_id: int,
    questions: List[Dict[str, Any]],
    student_name: Optional[str] = None,
    student_id: Optional[str] = None,
    language: str = "tt",
) -> bytes:
    """
    Renders an official A4 (210 x 297 mm) OCR test blank with ArUco markers,
    QR code, student name header, and 10x10 mm answer boxes.
    """
    fig = _create_blank_figure(
        assignment_id=assignment_id,
        title=title,
        variant_id=variant_id,
        questions=questions,
        student_name=student_name,
        student_id=student_id,
        language=language,
        dpi=300,
    )

    buf = io.BytesIO()
    FigureCanvasPdf(fig).print_pdf(buf)
    buf.seek(0)
    return buf.getvalue()


def render_batch_blanks_pdf(
    assignment_id: str,
    title: str,
    variants: List[Dict[str, Any]],
    students: List[Dict[str, str]],
    language: str = "tt",
) -> bytes:
    """
    Renders a multi-page PDF with one pre-named test blank per student in the class,
    automatically alternating variants across the roster.
    """
    buf = io.BytesIO()
    num_variants = max(1, len(variants))

    with PdfPages(buf) as pdf:
        target_students = students if students else [{"student_id": "", "full_name": None}]

        for idx, student in enumerate(target_students):
            var_idx = idx % num_variants
            var_obj = variants[var_idx] if variants else {}
            variant_id = var_obj.get("variant_id", var_idx + 1)
            questions = var_obj.get("questions", [])
            stu_name = student.get("full_name") or student.get("student_name")
            stu_id = student.get("student_id") or None

            fig = _create_blank_figure(
                assignment_id=assignment_id,
                title=title,
                variant_id=variant_id,
                questions=questions,
                student_name=stu_name,
                student_id=stu_id,
                language=language,
                dpi=300,
            )
            FigureCanvasPdf(fig)
            pdf.savefig(fig, dpi=300, bbox_inches=None)

    buf.seek(0)
    return buf.getvalue()
