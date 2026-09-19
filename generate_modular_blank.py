"""
Generate a production-ready, compact modular Tatar OCR test blank PDF.
Features:
- Standard A4 (210 x 297 mm)
- 4 Page-corner ArUco markers for global perspective rectification (DICT_4X4_50, IDs 0, 1, 2, 3)
- Compact Header with QR code and Student Name / ID input cells
- 8 Modular Question Blocks fully utilizing the vertical space (no empty dead space!)
- Exactly 10x10 mm answer letter cells matching the Tatar OCR training dataset!
- Clean white quiet zones (>= 4mm) around all markers to prevent contour merging
- Zero text overlap
"""

import json
from pathlib import Path
import cv2
import numpy as np
import qrcode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

# Setup font for Tatar letters
font_path = Path("fonts/PlaypenSans.ttf")
if not font_path.exists():
    font_path = Path("fonts/Rubik-Regular.ttf")
if not font_path.exists():
    font_path = Path("C:/Windows/Fonts/arial.ttf")

font_name = "TatarFont"
fe = fm.FontEntry(fname=str(font_path), name=font_name)
fm.fontManager.ttflist.append(fe)

def generate_aruco_img(marker_id: int, size_px: int = 140, dict_type=cv2.aruco.DICT_4X4_50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    return marker_img

def generate_qr_img(payload_str: str, box_size: int = 10):
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

def create_blank_pdf(output_pdf="test_blank_sample.pdf", output_png="test_blank_sample.png"):
    page_w_mm, page_h_mm = 210.0, 297.0
    fig_w_in, fig_h_in = page_w_mm / 25.4, page_h_mm / 25.4
    
    # Fill 100% of the canvas with ZERO matplotlib margin waste
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w_mm)
    ax.set_ylim(0, page_h_mm)
    ax.invert_yaxis()
    ax.axis('off')
    
    # -------------------------------------------------------------
    # 1. Page Corner ArUco Markers (Printer-safe 6mm paper margin)
    # -------------------------------------------------------------
    corner_size_mm = 14.0
    margin_corner_mm = 6.0  # Safe for any office laser/inkjet printer (hardware margin is ~4-5mm)
    
    corner_positions = {
        0: (margin_corner_mm, margin_corner_mm),                               # TL
        1: (page_w_mm - margin_corner_mm - corner_size_mm, margin_corner_mm), # TR
        2: (margin_corner_mm, page_h_mm - margin_corner_mm - corner_size_mm), # BL
        3: (page_w_mm - margin_corner_mm - corner_size_mm, page_h_mm - margin_corner_mm - corner_size_mm), # BR
    }
    
    for cid, (cx, cy) in corner_positions.items():
        c_img = generate_aruco_img(cid, size_px=140)
        ax.imshow(c_img, cmap='gray', extent=(cx, cx + corner_size_mm, cy + corner_size_mm, cy), zorder=10)
    
    # -------------------------------------------------------------
    # 2. Header Section (Compact, Well-Spaced)
    # -------------------------------------------------------------
    header_top_y = 6.0
    header_left_x = 23.0
    
    # QR Code with Test Metadata
    qr_payload = json.dumps({
        "tid": "TAT-2026-Q1",
        "var": 1,
        "page": 1,
        "tot": 1,
        "n_q": 8
    }, separators=(',', ':'))
    
    qr_img = generate_qr_img(qr_payload)
    qr_size_mm = 18.0
    qr_x = header_left_x
    qr_y = header_top_y
    ax.imshow(qr_img, extent=(qr_x, qr_x + qr_size_mm, qr_y + qr_size_mm, qr_y), zorder=10)
    
    ax.text(qr_x + qr_size_mm / 2.0, qr_y + qr_size_mm + 2.5, "TAT-2026-Q1 (Var 1)",
            fontsize=5.5, fontname='sans-serif', ha='center', va='top', color='#475569')

    # Title & Subtitle
    title_x = qr_x + qr_size_mm + 6.0
    ax.text(title_x, header_top_y + 1.5, "«ДӘРЕСХАНӘ» ТАТАР ТЕЛЕ ТЕСТЫ",
            fontsize=12.0, fontname=font_name, ha='left', va='top', color='#0f172a')
    ax.text(title_x, header_top_y + 7.0, "Җавап бланкы • Һәр шакмакка 10×10 мм бер баш хәреф языгыз",
            fontsize=7.2, fontname=font_name, ha='left', va='top', color='#475569')

    # Student Identification Field (Name / Code)
    name_lbl_y = header_top_y + 12.0
    ax.text(title_x, name_lbl_y, "Фамилия, исем (яки укучы коды):",
            fontsize=7.2, fontname=font_name, ha='left', va='top', color='#1e293b')
    
    name_cell_w = 9.5  # 9.5x9.5 mm per name cell
    name_cell_h = 9.5
    num_name_cells = 15  # Fits up to 15 characters
    name_cells_start_x = title_x
    name_cells_start_y = name_lbl_y + 4.5
    
    for i in range(num_name_cells):
        nx = name_cells_start_x + i * name_cell_w
        rect = patches.Rectangle((nx, name_cells_start_y), name_cell_w, name_cell_h,
                                 linewidth=0.8, edgecolor='#000000', facecolor='none')
        ax.add_patch(rect)
        
    # Header separator
    header_sep_y = 35.0
    ax.plot([margin_corner_mm, page_w_mm - margin_corner_mm], [header_sep_y, header_sep_y], color='#cbd5e1', linewidth=0.8)
    
    # -------------------------------------------------------------
    # 3. Modular Question Blocks (8 Questions Filling Page Efficiently)
    # -------------------------------------------------------------
    questions = [
        {"id": 11, "title": "1) Куегыз сүзне юнәлеш килешендә: китап ->", "cells": 8},
        {"id": 12, "title": "2) Куегыз сүзне чыгыш килешендә: өстәл ->", "cells": 8},
        {"id": 13, "title": "3) Сүзгә күплек сан кушымчасын ялгагыз: бала ->", "cells": 8},
        {"id": 14, "title": "4) Тәрҗемә итегез (доброе утро): иртә ->", "cells": 7},
        {"id": 15, "title": "5) Куегыз сүзне иялек килешендә: урман ->", "cells": 9},
        {"id": 16, "title": "6) Антонимны (кире мәгънәне) табыгыз: ялган ->", "cells": 6},
        {"id": 17, "title": "7) Тәрҗемә итегез (спасибо): ->", "cells": 7},
        {"id": 18, "title": "8) Куегыз сүзне төшем килешендә: дәфтәр ->", "cells": 8},
    ]
    
    q_marker_size_mm = 11.0
    q_start_y = 40.0
    q_pitch_y = 28.0  # 28mm pitch: 8 questions * 28 = 224mm!
    
    cell_w_mm = 10.0  # Exactly 10mm x 10mm matching training dataset!
    cell_h_mm = 10.0
    
    for idx, q in enumerate(questions):
        block_y = q_start_y + idx * q_pitch_y
        
        # Left Margin ArUco Anchor (At x=6mm, matching corner markers)
        marker_x = 6.0
        marker_y = block_y + 4.5
        m_img = generate_aruco_img(q["id"], size_px=110)
        ax.imshow(m_img, cmap='gray',
                  extent=(marker_x, marker_x + q_marker_size_mm, marker_y + q_marker_size_mm, marker_y),
                  zorder=10)
        
        # Small question ID label under marker
        ax.text(marker_x + q_marker_size_mm / 2.0, marker_y + q_marker_size_mm + 2.0,
                f"ID {q['id']}", fontsize=5.5, fontname='sans-serif', ha='center', va='top', color='#64748b')
        
        # Question Prompt Text (Clear, starts at x=22mm, giving 5mm quiet zone after marker at 17mm)
        text_x = 22.0
        text_y = block_y + 1.5
        ax.text(text_x, text_y, q["title"], fontsize=8.8, fontname=font_name, ha='left', va='top', color='#0f172a')
        
        # Draw 10mm x 10mm Cells (Well separated below prompt)
        cells_x = 22.0
        cells_y = block_y + 9.5
        for c in range(q["cells"]):
            cx = cells_x + c * cell_w_mm
            rect = patches.Rectangle((cx, cells_y), cell_w_mm, cell_h_mm,
                                     linewidth=0.8, edgecolor='#000000', facecolor='none')
            ax.add_patch(rect)
            
        # Separator line between questions
        if idx < len(questions) - 1:
            sep_y = block_y + q_pitch_y - 2.5
            ax.plot([20.0, page_w_mm - 10.0], [sep_y, sep_y], color='#e2e8f0', linewidth=0.6, linestyle='--')
            
    # -------------------------------------------------------------
    # 4. Footer Instructions (Cleanly framed above bottom markers)
    # -------------------------------------------------------------
    footer_y = 267.0
    ax.plot([margin_corner_mm, page_w_mm - margin_corner_mm], [footer_y, footer_y], color='#cbd5e1', linewidth=0.8)
    
    inst_text = (
        "Игътибар: Бланкны бөгәргә ярамый. Җавапларны шакмак эченә төгәл баш хәрефләр белән генә языгыз.\n"
        "Татар хәрефләре: Ә, Җ, Ң, Ө, Ү, Һ. Меткалар (ArUco / QR) өстенә язмагыз."
    )
    ax.text(page_w_mm / 2.0, footer_y + 2.5, inst_text,
            fontsize=7.0, fontname=font_name, ha='center', va='top', color='#475569', multialignment='center')

    # Save PDF and PNG
    out_pdf_path = Path(output_pdf)
    out_png_path = Path(output_png)
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(str(out_pdf_path), format='pdf', bbox_inches=None)
    plt.savefig(str(out_png_path), format='png', dpi=300, bbox_inches=None)
    plt.close()
    print(f"Generated compact blank: {out_pdf_path} and {out_png_path}")

if __name__ == "__main__":
    create_blank_pdf()
