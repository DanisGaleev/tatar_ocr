import json
from pathlib import Path
import numpy as np
import pytest

from blank_pipeline import (
    TATAR_UPPERCASE,
    RUSSIAN_UPPERCASE,
    TATAR_SPECIFIC_LETTERS,
    build_alphabet_mask,
    apply_language_logit_mask,
    BlankOCRScanner,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT_DIR / "pipeline_spec.json"
DEFAULT_MODEL = ROOT_DIR / "models" / "tatar_ocr_v4_handwritten39_int8.onnx"


def _render_glyph_cell(ch: str) -> np.ndarray:
    """Render a synthetic 100x100 cell image containing a single printed Cyrillic character."""
    from PIL import Image, ImageDraw, ImageFont

    font_candidates = [
        ROOT_DIR / "fonts" / "Rubik-Regular.ttf",
        ROOT_DIR / "fonts" / "PlaypenSans.ttf",
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    font_path = next((p for p in font_candidates if p.exists()), None)
    img = Image.new("L", (100, 100), color=255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(font_path), 64) if font_path else ImageFont.load_default()
    draw.text((22, 14), ch, fill=20, font=font)
    arr = np.array(img, dtype=np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


class TestOCRRussianMaskSpecAndLogits:
    def test_pipeline_spec_russian_alphabet_alignment(self):
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        tt_alpha = spec["model"]["alphabet"]
        ru_alpha = spec["model"]["alphabets"]["ru"]

        assert len(tt_alpha) == 39
        assert len(ru_alpha) == 33
        assert set(ru_alpha).issubset(set(tt_alpha))
        assert set(tt_alpha) - set(ru_alpha) == {"Ә", "Ө", "Ү", "Җ", "Ң", "Һ"}
        assert TATAR_SPECIFIC_LETTERS == {"Ә", "Ө", "Ү", "Җ", "Ң", "Һ"}

    def test_logit_mask_zeroes_tatar_letters_and_renormalizes(self):
        mask = build_alphabet_mask(TATAR_UPPERCASE, RUSSIAN_UPPERCASE)
        assert mask.shape == (39,)
        assert int(mask.sum()) == 33

        # Deliberately put highest raw logits on Tatar-specific letters Ә (idx 1) and Ө (idx 19)
        logits = np.full((1, 39), -5.0, dtype=np.float32)
        logits[0, TATAR_UPPERCASE.index("Ә")] = 20.0
        logits[0, TATAR_UPPERCASE.index("Ө")] = 15.0
        logits[0, TATAR_UPPERCASE.index("А")] = 4.0
        logits[0, TATAR_UPPERCASE.index("О")] = 2.0

        probs_ru = apply_language_logit_mask(logits, TATAR_UPPERCASE, RUSSIAN_UPPERCASE)
        assert probs_ru.shape == (1, 39)
        assert np.isclose(float(probs_ru.sum()), 1.0, atol=1e-5)

        for tatar_ch in TATAR_SPECIFIC_LETTERS:
            assert probs_ru[0, TATAR_UPPERCASE.index(tatar_ch)] == 0.0

        best_idx = int(np.argmax(probs_ru[0]))
        assert TATAR_UPPERCASE[best_idx] == "А"


class TestOCRRussianMaskNativeRuntime:
    def test_native_scanner_enforces_russian_mask(self):
        if not DEFAULT_MODEL.exists():
            pytest.skip("ONNX model file not present")
        try:
            scanner = BlankOCRScanner(model_path=DEFAULT_MODEL, spec_path=SPEC_PATH)
        except ImportError as exc:
            pytest.skip(f"tatar_ocr_native not built: {exc}")

        # Verify mask API on native module
        ru_mask = scanner._native.build_language_mask("ru")
        assert len(ru_mask) == 39
        assert sum(ru_mask) == 33
        ru_alpha = scanner._native.alphabet_for_language("ru")
        assert len(ru_alpha) == 33
        assert all(ch not in TATAR_SPECIFIC_LETTERS for ch in ru_alpha)

        # Render both Russian letters and Tatar-only letters (Ә, Ө, Ү)
        for ch in ("А", "О", "У", "К", "Ә", "Ө", "Ү"):
            cell_bgr = _render_glyph_cell(ch)
            res_ru = scanner.classify_cell(cell_bgr, force_predict=True, language="ru")
            assert res_ru["char"] not in TATAR_SPECIFIC_LETTERS
            assert res_ru["char"] in RUSSIAN_UPPERCASE
            for cand_ch, _ in res_ru["top3"]:
                assert cand_ch not in TATAR_SPECIFIC_LETTERS
            for item in res_ru["top_items"]:
                assert item["char"] not in TATAR_SPECIFIC_LETTERS

        # Verify Tatar mode still recognizes Tatar-specific letters when requested
        cell_tatar = _render_glyph_cell("Ә")
        res_tt = scanner.classify_cell(cell_tatar, force_predict=True, language="tt")
        assert res_tt["char"] in TATAR_UPPERCASE

    def test_native_process_blank_russian_mode_emits_zero_tatar_letters(self):
        sample_blank = ROOT_DIR / "benchmark" / "images" / "blank_01.jpg"
        if not DEFAULT_MODEL.exists() or not sample_blank.exists():
            pytest.skip("Benchmark blank or ONNX model not present")
        try:
            scanner = BlankOCRScanner(model_path=DEFAULT_MODEL, spec_path=SPEC_PATH)
        except ImportError as exc:
            pytest.skip(f"tatar_ocr_native not built: {exc}")

        res_ru = scanner.process_image(sample_blank, language="ru")
        assert res_ru["status"] == "success"
        for q in res_ru["questions"]:
            for ch in q["text"]:
                assert ch not in TATAR_SPECIFIC_LETTERS
            for c in q["cells"]:
                if not c["is_empty"]:
                    assert c["char"] in RUSSIAN_UPPERCASE
                for top_ch, _ in c["top3"]:
                    if top_ch != " ":
                        assert top_ch in RUSSIAN_UPPERCASE
