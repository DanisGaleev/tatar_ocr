import sys
import base64
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np
import torch

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blank_pipeline import BlankOCRScanner
from tatar_ocr_dataset import TatarOCRNet, ALL_CHARS
from tatar_ocr_augmentation import TATAR_UPPERCASE


class OCRService:
    _instance: Optional["OCRService"] = None

    def __init__(self):
        self.device = torch.device("cpu")
        self.blank_scanner = BlankOCRScanner(model_path="models/finetuned_uppercase39.pth", device=self.device)
        self.model_39 = self.blank_scanner.model

        self.model_102: Optional[TatarOCRNet] = None
        weights_102 = PROJECT_ROOT / "models" / "finetuned_real_tatar_ocr.pth"
        if not weights_102.exists():
            weights_102 = PROJECT_ROOT / "models" / "best_tatar_ocr_net.pth"
        if weights_102.exists():
            try:
                m102 = TatarOCRNet(num_classes=len(ALL_CHARS)).to(self.device)
                m102.load_state_dict(torch.load(weights_102, map_location=self.device, weights_only=True))
                m102.eval()
                self.model_102 = m102
            except Exception as e:
                print(f"Warning: Could not load 102-class model: {e}")

    @classmethod
    def get_instance(cls) -> "OCRService":
        if cls._instance is None:
            cls._instance = OCRService()
        return cls._instance

    @staticmethod
    def decode_image_from_b64(image_b64: str) -> np.ndarray:
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(image_b64)
        return OCRService.decode_image_from_bytes(img_bytes)

    @staticmethod
    def decode_image_from_bytes(img_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from provided data.")
        return img_bgr

    def scan_blank(self, img_bgr: np.ndarray, annotate: bool = True) -> Dict[str, Any]:
        res = self.blank_scanner.process_blank(img_bgr, annotate=annotate)
        annotated_b64 = None
        if annotate and "annotated_bgr" in res and res["annotated_bgr"] is not None:
            resized_annotated = cv2.resize(res["annotated_bgr"], (1050, 1485))
            _, buf = cv2.imencode(".jpg", resized_annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")

        return {
            "status": "success",
            "rectification_method": res["rectification_method"],
            "student_name": res["student_name"],
            "questions": res["questions"],
            "stats": res["stats"],
            "annotated_b64": annotated_b64,
        }

    def predict_box(self, img_bytes: bytes, contrast_mode: str = "percentile") -> Dict[str, Any]:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image.")

        ch, cw = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        my = max(1, int(round(ch * 0.11)))
        mx = max(1, int(round(cw * 0.11)))
        inner_gray = gray[my:ch - my, mx:cw - mx]
        ih, iw = inner_gray.shape[:2]

        bg_val = float(np.median(inner_gray))
        ink_mask = (inner_gray < (bg_val - 22)).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask)

        valid_ink = np.zeros_like(ink_mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            touches_edge = (bx <= 1 or by <= 1 or (bx + bw) >= iw - 1 or (by + bh) >= ih - 1)
            is_edge_line = (
                (bw > 0.75 * iw and bh <= 5) or
                (bh > 0.75 * ih and bw <= 5) or
                (touches_edge and (bh > 3.5 * bw or bw > 3.5 * bh))
            )
            if area >= 20 and not is_edge_line:
                valid_ink[labels == i] = 1

        ys, xs = np.where(valid_ink > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = int(np.min(xs)), int(np.max(xs))
            y1, y2 = int(np.min(ys)), int(np.max(ys))
            pad = 2
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(iw - 1, x2 + pad)
            y2 = min(ih - 1, y2 + pad)
            glyph_crop = inner_gray[y1:y2 + 1, x1:x2 + 1]
        else:
            glyph_crop = inner_gray

        gh, gw = glyph_crop.shape[:2]
        max_dim = max(gh, gw)
        target_dim = 46.0
        scale = target_dim / float(max_dim + 1e-5)
        new_w = max(1, min(60, int(round(gw * scale))))
        new_h = max(1, min(60, int(round(gh * scale))))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        resized_glyph = cv2.resize(glyph_crop, (new_w, new_h), interpolation=interp)

        if contrast_mode == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            norm_glyph = clahe.apply(resized_glyph)
        else:
            p_lo = np.percentile(resized_glyph, 2)
            p_hi = np.percentile(resized_glyph, 98)
            norm_glyph = np.clip((resized_glyph.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-5) * 240.0 + 10.0, 0, 255).astype(np.uint8)

        inp_64 = np.full((64, 64), 250, dtype=np.uint8)
        off_x = (64 - new_w) // 2
        off_y = (64 - new_h) // 2
        inp_64[off_y:off_y + new_h, off_x:off_x + new_w] = norm_glyph

        tensor = (torch.from_numpy(inp_64.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0) - 0.5) / 0.5

        def get_top5(m, chars):
            with torch.no_grad():
                probs = torch.softmax(m(tensor), dim=1)[0]
            top_k = min(5, len(chars))
            top_probs, top_indices = torch.topk(probs, k=top_k)
            return [{
                "char": chars[i],
                "unicode": f"U+{ord(chars[i]):04X}",
                "confidence": round(float(p) * 100.0, 2)
            } for i, p in zip(top_indices.numpy(), top_probs.numpy())]

        def to_b64(im):
            _, buf = cv2.imencode(".png", im)
            return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")

        res_dict = {
            "stages": {
                "cropped": to_b64(img_bgr),
                "padded": to_b64(resized_glyph),
                "normalized_64x64": to_b64(inp_64)
            },
            "predictions_39": get_top5(self.model_39, TATAR_UPPERCASE),
        }
        if self.model_102 is not None:
            res_dict["predictions_102"] = get_top5(self.model_102, ALL_CHARS)
        return res_dict
