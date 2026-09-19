import sys
import os
import json
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

WORKSPACE_DIR = Path(r"c:\Users\galee\PycharmProjects\tatar_ocr")
sys.path.insert(0, str(WORKSPACE_DIR))
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import torch
from PIL import Image

from tatar_ocr_dataset import TatarOCRNet, ALL_CHARS
from tatar_ocr_augmentation import TATAR_UPPERCASE

PORT = 7860
DEVICE = torch.device("cpu")

print("Loading OCR models...")
# 1. 39-Class Model
model_39 = TatarOCRNet(num_classes=len(TATAR_UPPERCASE)).to(DEVICE)
weights_39_path = WORKSPACE_DIR / "models" / "finetuned_uppercase39.pth"
if not weights_39_path.exists():
    weights_39_path = WORKSPACE_DIR / "models" / "baseline_synthetic_uppercase39.pth"
if weights_39_path.exists():
    model_39.load_state_dict(torch.load(weights_39_path, map_location=DEVICE, weights_only=True))
    model_39.eval()
    print(f"Loaded 39-class model from {weights_39_path}")

# 2. 102-Class Model
model_102 = TatarOCRNet(num_classes=len(ALL_CHARS)).to(DEVICE)
weights_102_path = WORKSPACE_DIR / "models" / "finetuned_real_tatar_ocr.pth"
if not weights_102_path.exists():
    weights_102_path = WORKSPACE_DIR / "models" / "best_tatar_ocr_net.pth"
if weights_102_path.exists():
    model_102.load_state_dict(torch.load(weights_102_path, map_location=DEVICE, weights_only=True))
    model_102.eval()
    print(f"Loaded 102-class model from {weights_102_path}")


def process_cropped_area(img_bytes, crop_box=None, rotation_deg=0, contrast_mode="percentile"):
    """
    Applies user rotation, extracts the crop box (x, y, w, h normalized or pixels),
    pads to square, applies contrast normalization, resizes to 64x64, and runs inference.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image.")

    h_orig, w_orig = img_bgr.shape[:2]

    # 1. Rotate if specified
    if rotation_deg != 0:
        center = (w_orig / 2.0, h_orig / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
        # Calculate new bounding dimensions
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h_orig * sin) + (w_orig * cos))
        new_h = int((h_orig * cos) + (w_orig * sin))
        rot_mat[0, 2] += (new_w / 2.0) - center[0]
        rot_mat[1, 2] += (new_h / 2.0) - center[1]
        img_bgr = cv2.warpAffine(img_bgr, rot_mat, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
        h_orig, w_orig = img_bgr.shape[:2]

    # 2. Extract Crop Box
    if crop_box:
        # crop_box is {x, y, width, height} in pixels of rotated image
        x = max(0, min(w_orig - 1, int(crop_box.get('x', 0))))
        y = max(0, min(h_orig - 1, int(crop_box.get('y', 0))))
        cw = max(4, min(w_orig - x, int(crop_box.get('width', w_orig))))
        ch = max(4, min(h_orig - y, int(crop_box.get('height', h_orig))))
        cropped = img_bgr[y:y+ch, x:x+cw]
    else:
        cropped = img_bgr

    # 3. Square Padding
    ch, cw = cropped.shape[:2]
    sq_size = max(ch, cw)
    edge_pixels = np.concatenate([
        cropped[0, :, :], cropped[-1, :, :],
        cropped[:, 0, :], cropped[:, -1, :]
    ], axis=0)
    bg_color = np.median(edge_pixels, axis=0).astype(np.uint8)

    padded = np.full((sq_size, sq_size, 3), bg_color, dtype=np.uint8)
    py = (sq_size - ch) // 2
    px = (sq_size - cw) // 2
    padded[py:py + ch, px:px + cw] = cropped

    # 4. Grayscale & Contrast Normalization
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    if contrast_mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        norm_gray = clahe.apply(gray)
    else:
        p_lo = np.percentile(gray, 2)
        p_hi = np.percentile(gray, 98)
        norm_gray = np.clip((gray.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-5) * 240.0 + 10.0, 0, 255).astype(np.uint8)

    # 5. Model Input 64x64
    inp_64 = cv2.resize(norm_gray, (64, 64), interpolation=cv2.INTER_AREA)

    # 6. Inference for both models
    tensor = (torch.from_numpy(inp_64.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0) - 0.5) / 0.5

    def get_top5(model, chars_list):
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
        top_k = min(5, len(chars_list))
        top_probs, top_indices = torch.topk(probs, k=top_k)
        res = []
        for idx, p in zip(top_indices.numpy(), top_probs.numpy()):
            c = chars_list[idx]
            res.append({
                "char": c,
                "unicode": f"U+{ord(c):04X}",
                "confidence": round(float(p) * 100.0, 2)
            })
        return res

    top5_39 = get_top5(model_39, TATAR_UPPERCASE)
    top5_102 = get_top5(model_102, ALL_CHARS)

    def to_b64(im):
        _, buf = cv2.imencode(".png", im)
        return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")

    return {
        "stages": {
            "cropped": to_b64(cropped),
            "padded": to_b64(padded),
            "normalized_64x64": to_b64(inp_64)
        },
        "predictions_39": top5_39,
        "predictions_102": top5_102
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Tatar OCR Interactive Box & Image Adjuster</title>
    <!-- Cropper.js for full drag, resize, zoom, rotate box controls -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.1/cropper.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.1/cropper.min.js"></script>
    <style>
        :root {
            --bg: #0b1120;
            --card-bg: #1e293b;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --border: #334155;
            --success: #22c55e;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 18px; min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
        h1 { font-size: 22px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
        .sub { color: var(--text-dim); font-size: 13px; }
        
        .layout { display: grid; grid-template-columns: 1fr 420px; gap: 20px; }
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
        
        /* Cropper Area */
        .canvas-container {
            height: 520px;
            background: #020617;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px dashed var(--border);
        }
        .canvas-container img { max-width: 100%; max-height: 100%; display: block; }
        
        /* Toolbar */
        .toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; align-items: center; }
        .btn {
            background: #334155;
            color: var(--text);
            border: 1px solid #475569;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s;
        }
        .btn:hover { background: var(--accent); border-color: var(--accent); }
        .btn-primary { background: var(--accent); border-color: var(--accent); }
        .btn-primary:hover { background: var(--accent-hover); }

        .slider-group { display: flex; align-items: center; gap: 8px; background: #0f172a; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border); }
        .slider-group label { font-size: 12px; color: var(--text-dim); white-space: nowrap; }
        .slider-group input[type="range"] { width: 100px; accent-color: var(--accent); }
        
        /* Samples bar */
        .samples-bar { display: flex; gap: 6px; margin-top: 10px; align-items: center; }
        .samples-bar span { font-size: 12px; color: var(--text-dim); }
        .btn-chip { background: #1e293b; border: 1px solid var(--border); color: #cbd5e1; padding: 4px 10px; border-radius: 4px; font-size: 12px; cursor: pointer; }
        .btn-chip:hover { background: var(--accent); color: #fff; }

        /* Right Column */
        .section-title { font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-dim); margin-bottom: 10px; }
        
        .stages-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 18px; }
        .stage-card {
            background: #0f172a;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px;
            text-align: center;
        }
        .stage-card img {
            width: 100%;
            height: 90px;
            object-fit: contain;
            background: #1e293b;
            border-radius: 4px;
            image-rendering: pixelated;
        }
        .stage-card .lbl { font-size: 11px; font-weight: 600; color: var(--text-dim); margin-top: 6px; }
        .stage-card.highlight { border-color: var(--accent); }
        .stage-card.highlight .lbl { color: var(--accent); }

        /* Predictions */
        .model-tab { display: flex; gap: 4px; margin-bottom: 12px; background: #0f172a; padding: 4px; border-radius: 8px; border: 1px solid var(--border); }
        .tab-btn { flex: 1; padding: 6px; font-size: 12px; font-weight: 600; border: none; background: transparent; color: var(--text-dim); border-radius: 6px; cursor: pointer; }
        .tab-btn.active { background: var(--card-bg); color: var(--text); }

        .pred-item {
            display: flex;
            align-items: center;
            gap: 14px;
            background: #0f172a;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
            transition: all 0.2s;
        }
        .pred-item.top { border-color: var(--success); background: rgba(34, 197, 94, 0.08); }
        .pred-glyph { font-size: 32px; font-weight: 700; width: 44px; text-align: center; }
        .pred-item.top .pred-glyph { color: var(--success); }
        .pred-body { flex: 1; }
        .pred-header { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
        .bar-wrap { background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden; }
        .bar-val { height: 100%; background: var(--accent); border-radius: 4px; transition: width 0.25s ease-out; }
        .pred-item.top .bar-val { background: var(--success); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Tatar OCR: Interactive Box & Image Adjuster</h1>
                <div class="sub">Move, resize the crop box, drag/pan the image, zoom, or rotate. Real-time preprocessing & model inference update instantly!</div>
            </div>
            <div>
                <input type="file" id="fileInput" accept="image/*" style="display:none">
                <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">Upload Image</button>
            </div>
        </header>

        <div class="layout">
            <!-- Left: Interactive Canvas -->
            <div class="card">
                <div class="canvas-container">
                    <img id="cropperImage" src="/sample/strip2_rot90_cw.png" alt="Crop Target">
                </div>

                <!-- Controls Toolbar -->
                <div class="toolbar">
                    <button class="btn" onclick="cropper.rotate(-90)" title="Rotate 90 Left">↺ 90°</button>
                    <button class="btn" onclick="cropper.rotate(90)" title="Rotate 90 Right">↻ 90°</button>
                    
                    <div class="slider-group">
                        <label>Fine Angle:</label>
                        <input type="range" id="angleSlider" min="-45" max="45" value="0" oninput="onFineRotate(this.value)">
                        <span id="angleVal" style="font-size: 11px; width: 28px; text-align: right;">0°</span>
                    </div>

                    <button class="btn" onclick="cropper.zoom(0.15)" title="Zoom In">+</button>
                    <button class="btn" onclick="cropper.zoom(-0.15)" title="Zoom Out">−</button>
                    <button class="btn" onclick="cropper.reset(); document.getElementById('angleSlider').value=0; document.getElementById('angleVal').innerText='0°';" title="Reset Box & Zoom">Reset</button>

                    <div class="slider-group" style="margin-left: auto;">
                        <label>Contrast:</label>
                        <select id="contrastSelect" onchange="triggerInference()" style="background:#1e293b; color:#fff; border:none; font-size:12px; padding:2px 6px; border-radius:4px;">
                            <option value="percentile" selected>Percentile Min/Max</option>
                            <option value="clahe">Adaptive CLAHE</option>
                        </select>
                    </div>
                </div>

                <div class="samples-bar">
                    <span>Quick Load:</span>
                    <button class="btn-chip" onclick="loadUrl('/sample/strip2_rot90_cw.png')">Full 6-Cell Strip</button>
                    <button class="btn-chip" onclick="loadUrl('/sample/crop_raw_1.png')">Ә (Cell 1)</button>
                    <button class="btn-chip" onclick="loadUrl('/sample/crop_raw_2.png')">А (Cell 2)</button>
                    <button class="btn-chip" onclick="loadUrl('/sample/crop_raw_3.png')">Б (Cell 3)</button>
                    <button class="btn-chip" onclick="loadUrl('/sample/crop_raw_4.png')">Җ (Cell 4)</button>
                    <button class="btn-chip" onclick="loadUrl('/sample/crop_raw_5.png')">Щ (Cell 5)</button>
                    <button class="btn-chip" onclick="loadUrl('/sample/crop_raw_6.png')">Ч (Cell 6)</button>
                </div>
            </div>

            <!-- Right: Exact Preprocessing Stages & Predictions -->
            <div class="card">
                <div class="section-title">Exact Preprocessed Input to Model</div>
                <div class="stages-row">
                    <div class="stage-card">
                        <img id="imgCropped" src="" alt="Cropped">
                        <div class="lbl">1. User Crop Box</div>
                    </div>
                    <div class="stage-card">
                        <img id="imgPadded" src="" alt="Padded">
                        <div class="lbl">2. Square Padded</div>
                    </div>
                    <div class="stage-card highlight">
                        <img id="imgNorm64" src="" alt="64x64">
                        <div class="lbl">3. CNN Inp 64x64</div>
                    </div>
                </div>

                <div class="section-title" style="display: flex; justify-content: space-between; align-items: center;">
                    <span>Model Predictions</span>
                    <span id="latencyTag" style="font-size: 11px; color: var(--success); font-weight: normal;">Live</span>
                </div>

                <div class="model-tab">
                    <button class="tab-btn active" id="tab39" onclick="switchModel('39')">39-Class Uppercase (Recommended)</button>
                    <button class="tab-btn" id="tab102" onclick="switchModel('102')">102-Class Full Model</button>
                </div>

                <div id="predictionsList">
                    <!-- Populated via JS -->
                </div>
            </div>
        </div>
    </div>

    <script>
        let cropper = null;
        let activeModel = '39';
        let latestPredictions = { predictions_39: [], predictions_102: [] };
        let debounceTimer = null;
        let baseRotation = 0;

        const imgElement = document.getElementById('cropperImage');
        const fileInput = document.getElementById('fileInput');

        function initCropper() {
            if (cropper) cropper.destroy();
            cropper = new Cropper(imgElement, {
                viewMode: 1,
                dragMode: 'move',
                autoCrop: true,
                autoCropArea: 0.35,
                restore: false,
                guides: true,
                center: true,
                highlight: true,
                cropBoxMovable: true,
                cropBoxResizable: true,
                toggleDragModeOnDblclick: true,
                ready() {
                    triggerInference();
                },
                crop(event) {
                    // Debounced real-time update during drag/resize
                    clearTimeout(debounceTimer);
                    debounceTimer = setTimeout(triggerInference, 120);
                }
            });
        }

        window.addEventListener('DOMContentLoaded', () => {
            initCropper();
        });

        // Paste from clipboard support
        window.addEventListener('paste', (e) => {
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            for (let item of items) {
                if (item.type.indexOf('image') !== -1) {
                    const blob = item.getAsFile();
                    loadBlob(blob);
                    break;
                }
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                loadBlob(e.target.files[0]);
            }
        });

        function loadBlob(blob) {
            const reader = new FileReader();
            reader.onload = function(evt) {
                imgElement.src = evt.target.result;
                initCropper();
            };
            reader.readAsDataURL(blob);
        }

        function loadUrl(url) {
            imgElement.src = url;
            initCropper();
        }

        function onFineRotate(val) {
            document.getElementById('angleVal').innerText = val + '°';
            if (cropper) {
                cropper.rotateTo(parseInt(val));
            }
        }

        function triggerInference() {
            if (!cropper) return;
            // Get cropped canvas directly from cropper at high resolution
            const cropCanvas = cropper.getCroppedCanvas({
                imageSmoothingEnabled: true,
                imageSmoothingQuality: 'high'
            });
            if (!cropCanvas) return;

            const cropB64 = cropCanvas.toDataURL('image/png').split(',')[1];
            const contrastMode = document.getElementById('contrastSelect').value;

            const t0 = performance.now();
            fetch('/api/predict_box', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_b64: cropB64,
                    contrast_mode: contrastMode
                })
            })
            .then(r => r.json())
            .then(data => {
                const elapsed = (performance.now() - t0).toFixed(0);
                document.getElementById('latencyTag').innerText = elapsed + ' ms';

                // Display pipeline stages
                document.getElementById('imgCropped').src = data.stages.cropped;
                document.getElementById('imgPadded').src = data.stages.padded;
                document.getElementById('imgNorm64').src = data.stages.normalized_64x64;

                latestPredictions = data;
                renderPredictions();
            })
            .catch(err => console.error(err));
        }

        function switchModel(mode) {
            activeModel = mode;
            document.getElementById('tab39').classList.toggle('active', mode === '39');
            document.getElementById('tab102').classList.toggle('active', mode === '102');
            renderPredictions();
        }

        function renderPredictions() {
            const list = document.getElementById('predictionsList');
            list.innerHTML = '';
            const preds = activeModel === '39' ? latestPredictions.predictions_39 : latestPredictions.predictions_102;
            if (!preds || !preds.length) return;

            preds.forEach((p, idx) => {
                const isTop = idx === 0;
                const div = document.createElement('div');
                div.className = 'pred-item ' + (isTop ? 'top' : '');
                div.innerHTML = `
                    <div class="pred-glyph">${p.char}</div>
                    <div class="pred-body">
                        <div class="pred-header">
                            <span><strong>Rank #${idx + 1}: '${p.char}'</strong> (${p.unicode})</span>
                            <span><strong>${p.confidence.toFixed(1)}%</strong></span>
                        </div>
                        <div class="bar-wrap">
                            <div class="bar-val" style="width: ${Math.max(2, p.confidence)}%;"></div>
                        </div>
                    </div>
                `;
                list.appendChild(div);
            });
        }
    </script>
</body>
</html>
"""

class OCRRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path.startswith("/sample/"):
            filename = os.path.basename(self.path)
            sample_path = WORKSPACE_DIR / "dataset" / "six_cells_test" / filename
            if not sample_path.exists():
                sample_path = WORKSPACE_DIR / filename
            if not sample_path.exists():
                sample_path = Path(r"C:\Users\galee\.gemini\antigravity\brain\e6d31b5c-5d27-456f-ade7-c736653ad00b") / filename
            if sample_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(sample_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/predict_box":
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len)
            try:
                data = json.loads(post_body.decode('utf-8'))
                img_bytes = base64.b64decode(data['image_b64'])
                contrast_mode = data.get('contrast_mode', 'percentile')

                res = process_cropped_area(
                    img_bytes,
                    crop_box=None,
                    rotation_deg=0,
                    contrast_mode=contrast_mode
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def run_server():
    server_address = ('127.0.0.1', PORT)
    httpd = HTTPServer(server_address, OCRRequestHandler)
    print(f"\n=======================================================")
    print(f"  Tatar OCR Interactive Adjuster & Inspector is RUNNING!")
    print(f"  Open in your browser: http://127.0.0.1:{PORT}")
    print(f"=======================================================\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
