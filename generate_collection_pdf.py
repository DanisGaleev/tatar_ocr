"""
Minimalist Tatar OCR Handwriting Collection Sheet Generator
Maximized printable rows (1cm x 1cm boxes), no decorative overhead.
Fits 24 rows per page (17 boxes of 1cm x 1cm per row) across 4-5 pages.
"""

import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import matplotlib.font_manager as fm

# Register font for Tatar letters
font_path = Path("fonts/PlaypenSans.ttf")
if not font_path.exists():
    font_path = Path("C:/Windows/Fonts/arial.ttf")

font_name = "PlaypenSans"
if font_path.exists():
    fe = fm.FontEntry(fname=str(font_path), name=font_name)
    fm.fontManager.ttflist.append(fe)
else:
    font_name = "sans-serif"

# ---------------------------------------------------------------------------
# Letter Schedule (Total: 120 rows across 5 pages = 24 rows per page)
# Heavy emphasis on difficult Tatar glyphs & confusion pairs
# ---------------------------------------------------------------------------

LETTER_SCHEDULE = [
    # Top priority: Tatar specific glyphs (4-5 rows each)
    ("Ә", 5),
    ("Җ", 5),
    ("Ң", 5),
    ("Ө", 5),
    ("Ү", 5),
    ("Һ", 5),

    # High confusion Cyrillic pairs (4 rows each)
    ("Щ", 4),
    ("Ц", 4),
    ("Ч", 4),
    ("Ш", 4),
    ("Д", 4),
    ("Ю", 4),
    ("Б", 4),
    ("Э", 4),

    # Standard Cyrillic letters (2 rows each)
    ("А", 2),
    ("В", 2),
    ("Г", 2),
    ("Е", 2),
    ("Ё", 2),
    ("Ж", 2),
    ("З", 2),
    ("И", 2),
    ("Й", 2),
    ("К", 2),
    ("Л", 2),
    ("М", 2),
    ("Н", 2),
    ("О", 2),
    ("П", 2),
    ("Р", 2),
    ("С", 2),
    ("Т", 2),
    ("У", 2),
    ("Ф", 2),
    ("Х", 2),
    ("Ъ", 2),
    ("Ы", 2),
    ("Ь", 2),
    ("Я", 2),

    # Extra reinforcement for the most difficult Tatar letters to reach exactly 120 rows
    ("Ә", 2),
    ("Җ", 2),
    ("Ң", 2),
    ("Ө", 2),
    ("Щ", 2),
]

# Expand row items
all_rows = []
for char, count in LETTER_SCHEDULE:
    for _ in range(count):
        all_rows.append(char)

ROWS_PER_PAGE = 24
TOTAL_PAGES = 5  # 5 pages * 24 = 120 rows! (120 * 17 = 2,040 individual 1cm boxes)


def generate_collection_pdf(output_pdf="dataset/tatar_handwriting_collection_sheets.pdf"):
    out_file = Path(output_pdf)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # A4 dimensions in mm: 210 x 297 mm
    page_w_mm, page_h_mm = 210.0, 297.0
    fig_w_in, fig_h_in = page_w_mm / 25.4, page_h_mm / 25.4

    # Utilitarian geometry to maximize space
    margin_left_mm = 10.0
    margin_right_mm = 10.0
    margin_top_mm = 8.0
    
    prompt_w_mm = 12.0
    prompt_gap_mm = 3.0
    
    # 17 boxes of exactly 10mm x 10mm (1cm x 1cm)
    box_size_mm = 10.0
    num_boxes = 17
    
    row_height_mm = 10.0
    row_gap_mm = 1.8  # Minimal compact gap between rows

    with PdfPages(str(out_file)) as pdf:
        for page_idx in range(TOTAL_PAGES):
            page_num = page_idx + 1
            fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
            ax.set_xlim(0, page_w_mm)
            ax.set_ylim(0, page_h_mm)
            ax.invert_yaxis()
            ax.axis('off')

            # Minimal single-line page marker at very top edge
            ax.text(
                margin_left_mm, 5.5,
                f"«Дәресханә» Бланк рукописных букв (клетка 1×1 см) • Лист {page_num}/{TOTAL_PAGES}",
                fontsize=7.0, fontname='sans-serif', color='#475569', va='bottom'
            )

            # Draw 24 rows
            page_start_row = page_idx * ROWS_PER_PAGE
            page_rows = all_rows[page_start_row : page_start_row + ROWS_PER_PAGE]

            for r_idx, char in enumerate(page_rows):
                y_top = margin_top_mm + r_idx * (row_height_mm + row_gap_mm)
                
                # 1. Letter prompt indicator on the left
                ax.add_patch(patches.Rectangle(
                    (margin_left_mm, y_top), prompt_w_mm, row_height_mm,
                    facecolor='#f8fafc', edgecolor='#000000', lw=0.8
                ))
                ax.text(
                    margin_left_mm + prompt_w_mm / 2.0, y_top + row_height_mm / 2.0,
                    char, fontname=font_name, fontsize=14, weight='normal',
                    ha='center', va='center', color='#000000'
                )

                # 2. 17 contiguous 1cm x 1cm boxes
                boxes_x_start = margin_left_mm + prompt_w_mm + prompt_gap_mm
                strip_w = num_boxes * box_size_mm
                
                # Outer rectangle of row strip
                ax.add_patch(patches.Rectangle(
                    (boxes_x_start, y_top), strip_w, row_height_mm,
                    facecolor='none', edgecolor='#000000', lw=0.7
                ))
                
                # Thin divider lines between boxes
                for b_idx in range(1, num_boxes):
                    x_div = boxes_x_start + b_idx * box_size_mm
                    ax.plot([x_div, x_div], [y_top, y_top + row_height_mm], color='#000000', lw=0.4)

            pdf.savefig(fig, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f"Generated Page {page_num}/{TOTAL_PAGES} ({len(page_rows)} rows)")

    print(f"\nPDF successfully saved to: {out_file.resolve().as_posix()}")
    return out_file


def render_preview_png(preview_png="dataset/collection_sheet_preview_page1.png"):
    out_png = Path(preview_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    page_w_mm, page_h_mm = 210.0, 297.0
    fig_w_in, fig_h_in = page_w_mm / 25.4, page_h_mm / 25.4
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    ax.set_xlim(0, page_w_mm)
    ax.set_ylim(0, page_h_mm)
    ax.invert_yaxis()
    ax.axis('off')

    margin_left_mm = 10.0
    margin_top_mm = 8.0
    prompt_w_mm = 12.0
    prompt_gap_mm = 3.0
    box_size_mm = 10.0
    num_boxes = 17
    row_height_mm = 10.0
    row_gap_mm = 1.8

    ax.text(
        margin_left_mm, 5.5,
        "«Дәресханә» Бланк рукописных букв (клетка 1×1 см) • Лист 1/5",
        fontsize=7.0, fontname='sans-serif', color='#475569', va='bottom'
    )

    page_rows = all_rows[0:ROWS_PER_PAGE]
    for r_idx, char in enumerate(page_rows):
        y_top = margin_top_mm + r_idx * (row_height_mm + row_gap_mm)
        ax.add_patch(patches.Rectangle(
            (margin_left_mm, y_top), prompt_w_mm, row_height_mm,
            facecolor='#f8fafc', edgecolor='#000000', lw=0.8
        ))
        ax.text(
            margin_left_mm + prompt_w_mm / 2.0, y_top + row_height_mm / 2.0,
            char, fontname=font_name, fontsize=14, weight='normal',
            ha='center', va='center', color='#000000'
        )

        boxes_x_start = margin_left_mm + prompt_w_mm + prompt_gap_mm
        strip_w = num_boxes * box_size_mm
        ax.add_patch(patches.Rectangle(
            (boxes_x_start, y_top), strip_w, row_height_mm,
            facecolor='none', edgecolor='#000000', lw=0.7
        ))
        for b_idx in range(1, num_boxes):
            x_div = boxes_x_start + b_idx * box_size_mm
            ax.plot([x_div, x_div], [y_top, y_top + row_height_mm], color='#000000', lw=0.4)

    fig.savefig(str(out_png), dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"Preview PNG saved to: {out_png}")


if __name__ == "__main__":
    generate_collection_pdf("dataset/tatar_handwriting_collection_sheets.pdf")
    render_preview_png("dataset/collection_sheet_preview_page1.png")
