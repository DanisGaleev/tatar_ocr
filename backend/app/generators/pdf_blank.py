import io
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import qrcode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

# Setup font for Tatar Cyrillic characters
_FONT_PATH = Path("fonts/Rubik-Regular.ttf")
if not _FONT_PATH.exists():
    _FONT_PATH = Path("fonts/PlaypenSans.ttf")
if not _FONT_PATH.exists():
    _FONT_PATH = Path("C:/Windows/Fonts/segoeui.ttf")
if not _FONT_PATH.exists():
    _FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")

_FONT_NAME = "TatarFont"
if _FONT_PATH.exists():
    _fe = fm.FontEntry(fname=str(_FONT_PATH.resolve()), name=_FONT_NAME)
    fm.fontManager.ttflist.append(_fe)
else:
    _FONT_NAME = "sans-serif"


def generate_aruco_img(marker_id: int, size_px: int = 140, dict_type=cv2.aruco.DICT_4X4_50) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
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


def render_blank_pdf(
    assignment_id: str,
    title: str,
    variant_id: int,
    questions: List[Dict[str, Any]],
    student_name: Optional[str] = None,
) -> bytes:
    """
    Renders an official A4 (210 x 297 mm) Tatar OCR test blank with ArUco markers,
    QR code, student name header, and 10x10 mm answer boxes.
    """
    page_w_mm, page_h_mm = 210.0, 297.0
    fig_w_in, fig_h_in = page_w_mm / 25.4, page_h_mm / 25.4

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w_mm)
    ax.set_ylim(0, page_h_mm)
    ax.invert_yaxis()
    ax.axis('off')

    # 1. Page Corner ArUco Markers (Printer-safe 6mm paper margin)
    corner_size_mm = 14.0
    margin_corner_mm = 6.0

    corner_positions = {
        0: (margin_corner_mm, margin_corner_mm),                               # TL
        1: (page_w_mm - margin_corner_mm - corner_size_mm, margin_corner_mm), # TR
        2: (margin_corner_mm, page_h_mm - margin_corner_mm - corner_size_mm), # BL
        3: (page_w_mm - margin_corner_mm - corner_size_mm, page_h_mm - margin_corner_mm - corner_size_mm), # BR
    }

    for cid, (cx, cy) in corner_positions.items():
        c_img = generate_aruco_img(cid, size_px=140)
        ax.imshow(c_img, cmap='gray', extent=(cx, cx + corner_size_mm, cy + corner_size_mm, cy), zorder=10)

    # 2. Header Section
    header_top_y = 6.0
    header_left_x = 23.0

    # QR Code with Test Metadata
    qr_payload = json.dumps({
        "tid": assignment_id,
        "var": variant_id,
        "page": 1,
        "tot": 1,
        "n_q": len(questions),
    }, separators=(',', ':'))

    qr_img = generate_qr_img(qr_payload)
    qr_size_mm = 18.0
    qr_x = header_left_x
    qr_y = header_top_y
    ax.imshow(qr_img, extent=(qr_x, qr_x + qr_size_mm, qr_y + qr_size_mm, qr_y), zorder=10)

    ax.text(
        qr_x + qr_size_mm / 2.0, qr_y + qr_size_mm + 2.5,
        f"{assignment_id} (Вар. {variant_id})",
        fontsize=5.5, fontname='sans-serif', ha='center', va='top', color='#475569'
    )

    # Title & Subtitle
    title_x = qr_x + qr_size_mm + 6.0
    display_title = title if len(title) <= 45 else title[:42] + "..."
    ax.text(title_x, header_top_y + 1.5, f"«ДӘРЕСХАНӘ» • {display_title.upper()}",
            fontsize=10.5, fontname=_FONT_NAME, ha='left', va='top', color='#0f172a', weight='bold')
    ax.text(title_x, header_top_y + 6.8, f"Җавап бланкы • Вариант {variant_id} • Һәр шакмакка 10×10 мм баш хәреф языгыз",
            fontsize=7.2, fontname=_FONT_NAME, ha='left', va='top', color='#475569')

    # Student Identification Field
    name_lbl_y = header_top_y + 11.5
    ax.text(title_x, name_lbl_y, "Фамилия, исем (укучы коды):",
            fontsize=7.2, fontname=_FONT_NAME, ha='left', va='top', color='#1e293b')

    name_cell_w = 9.0
    name_cell_h = 9.0
    num_name_cells = 16
    name_cells_start_x = title_x
    name_cells_start_y = name_lbl_y + 4.2

    name_letters = list(student_name.upper()) if student_name else []

    for i in range(num_name_cells):
        nx = name_cells_start_x + i * name_cell_w
        rect = patches.Rectangle((nx, name_cells_start_y), name_cell_w, name_cell_h,
                                 linewidth=0.8, edgecolor='#000000', facecolor='none')
        ax.add_patch(rect)
        if i < len(name_letters) and name_letters[i] != ' ':
            ax.text(nx + name_cell_w / 2.0, name_cells_start_y + name_cell_h / 2.0,
                    name_letters[i], fontsize=8.0, fontname=_FONT_NAME,
                    ha='center', va='center', color='#0f172a', weight='bold')

    # Header separator
    header_sep_y = 34.5
    ax.plot([margin_corner_mm, page_w_mm - margin_corner_mm], [header_sep_y, header_sep_y],
            color='#cbd5e1', linewidth=0.8)

    # 3. Modular Question Blocks (Up to 8 Questions)
    q_marker_size_mm = 11.0
    q_start_y = 38.0
    q_pitch_y = 28.0
    cell_w_mm = 10.0
    cell_h_mm = 10.0

    for idx, q in enumerate(questions[:8]):
        block_y = q_start_y + idx * q_pitch_y
        marker_id = q.get("marker_id", 11 + idx)

        # Left Margin ArUco Anchor
        marker_x = 6.0
        marker_y = block_y + 4.0
        m_img = generate_aruco_img(marker_id, size_px=110)
        ax.imshow(m_img, cmap='gray',
                  extent=(marker_x, marker_x + q_marker_size_mm, marker_y + q_marker_size_mm, marker_y),
                  zorder=10)

        # Marker ID label
        ax.text(marker_x + q_marker_size_mm / 2.0, marker_y + q_marker_size_mm + 1.8,
                f"ID {marker_id}", fontsize=5.5, fontname='sans-serif', ha='center', va='top', color='#64748b')

        # Question Prompt Text & Dynamic Layout
        import textwrap
        prompt_text = q.get("prompt", q.get("prompt_tt", f"Сорау №{idx+1}"))
        q_prefix = f"{idx+1}) "

        text_x = 22.0
        text_y = block_y + 1.0

        # Handle multi-line prompts or wrap long single-line prompts
        raw_lines = prompt_text.split("\n")
        lines = []
        for rl in raw_lines:
            if len(rl) > 75:
                lines.extend(textwrap.wrap(rl, width=75))
            else:
                lines.append(rl)

        if len(lines) > 1:
            q_full_text = q_prefix + lines[0] + "\n" + "   " + "\n   ".join(lines[1:3])
            font_size = 6.8 if len(lines) > 2 or any(len(l) > 65 for l in lines) else 7.4
            cells_y = block_y + 12.0
        else:
            q_full_text = q_prefix + lines[0]
            font_size = 8.2 if len(q_full_text) <= 65 else 7.2
            cells_y = block_y + 9.5

        ax.text(text_x, text_y, q_full_text, fontsize=font_size, fontname=_FONT_NAME,
                ha='left', va='top', color='#0f172a')

        # 10mm x 10mm Answer Cells
        cells_x = 22.0
        cell_count = max(1, min(q.get("cell_count", 8), 12))

        if cell_count == 1:
            ax.text(cells_x - 1.5, cells_y + cell_h_mm / 2.0, "Җавап:",
                    fontsize=6.5, fontname=_FONT_NAME, ha='right', va='center', color='#64748b')

        for c in range(cell_count):
            cx = cells_x + c * cell_w_mm
            rect = patches.Rectangle((cx, cells_y), cell_w_mm, cell_h_mm,
                                     linewidth=0.8, edgecolor='#000000', facecolor='none')
            ax.add_patch(rect)
            # Subtle cell index number in top-left
            ax.text(cx + 1.2, cells_y + 1.0, str(c + 1),
                    fontsize=4.2, fontname='sans-serif', ha='left', va='top', color='#94a3b8')

        # Separator line between questions
        if idx < min(len(questions), 8) - 1:
            sep_y = block_y + q_pitch_y - 2.5
            ax.plot([20.0, page_w_mm - 10.0], [sep_y, sep_y], color='#e2e8f0', linewidth=0.6, linestyle='--')

    # 4. Footer Instructions
    footer_y = 267.0
    ax.plot([margin_corner_mm, page_w_mm - margin_corner_mm], [footer_y, footer_y], color='#cbd5e1', linewidth=0.8)

    inst_text = (
        "Игътибар: Бланкны бөгәргә ярамый. Җавапларны шакмак эченә төгәл баш хәрефләр белән генә языгыз.\n"
        "Татар хәрефләре: Ә, Җ, Ң, Ө, Ү, Һ. Меткалар (ArUco / QR) өстенә язмагыз."
    )
    ax.text(page_w_mm / 2.0, footer_y + 2.5, inst_text,
            fontsize=7.0, fontname=_FONT_NAME, ha='center', va='top', color='#475569', multialignment='center')

    # Render to PDF in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='pdf', bbox_inches=None)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
