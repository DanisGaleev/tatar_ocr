import sys
import os
import io
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
from blank_pipeline import BlankOCRScanner

PORT = 7860
DEVICE = torch.device("cpu")

print("Initializing Tatar OCR Inspector Server & Blank Scanner...")

# 1. Models
model_39 = TatarOCRNet(num_classes=len(TATAR_UPPERCASE)).to(DEVICE)
weights_39_path = WORKSPACE_DIR / "models" / "finetuned_uppercase39.pth"
if not weights_39_path.exists():
    weights_39_path = WORKSPACE_DIR / "models" / "baseline_synthetic_uppercase39.pth"
if weights_39_path.exists():
    model_39.load_state_dict(torch.load(weights_39_path, map_location=DEVICE, weights_only=True))
    model_39.eval()
    print(f"Loaded 39-class model from {weights_39_path}")

model_102 = TatarOCRNet(num_classes=len(ALL_CHARS)).to(DEVICE)
weights_102_path = WORKSPACE_DIR / "models" / "finetuned_real_tatar_ocr.pth"
if not weights_102_path.exists():
    weights_102_path = WORKSPACE_DIR / "models" / "best_tatar_ocr_net.pth"
if weights_102_path.exists():
    model_102.load_state_dict(torch.load(weights_102_path, map_location=DEVICE, weights_only=True))
    model_102.eval()
    print(f"Loaded 102-class model from {weights_102_path}")

# 2. Blank Scanner
blank_scanner = BlankOCRScanner(model_path="models/finetuned_uppercase39.pth", device=DEVICE)


def process_cropped_area(img_bytes, contrast_mode="percentile"):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image.")

    ch, cw = img_bgr.shape[:2]
    sq_size = max(ch, cw)
    edge_pixels = np.concatenate([
        img_bgr[0, :, :], img_bgr[-1, :, :],
        img_bgr[:, 0, :], img_bgr[:, -1, :]
    ], axis=0)
    bg_color = np.median(edge_pixels, axis=0).astype(np.uint8)

    padded = np.full((sq_size, sq_size, 3), bg_color, dtype=np.uint8)
    py = (sq_size - ch) // 2
    px = (sq_size - cw) // 2
    padded[py:py + ch, px:px + cw] = img_bgr

    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    if contrast_mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        norm_gray = clahe.apply(gray)
    else:
        p_lo = np.percentile(gray, 2)
        p_hi = np.percentile(gray, 98)
        norm_gray = np.clip((gray.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-5) * 240.0 + 10.0, 0, 255).astype(np.uint8)

    inp_64 = cv2.resize(norm_gray, (64, 64), interpolation=cv2.INTER_AREA)
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

    return {
        "stages": {
            "cropped": to_b64(img_bgr),
            "padded": to_b64(padded),
            "normalized_64x64": to_b64(inp_64)
        },
        "predictions_39": get_top5(model_39, TATAR_UPPERCASE),
        "predictions_102": get_top5(model_102, ALL_CHARS)
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Tatar OCR Suite: Full Blank Scanner & Character Inspector</title>
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
            --warn: #eab308;
            --danger: #ef4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 18px; min-height: 100vh; }
        .container { max-width: 1560px; margin: 0 auto; }
        
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; border-bottom: 1px solid var(--border); padding-bottom: 12px; }
        h1 { font-size: 22px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
        .nav-tabs { display: flex; gap: 8px; }
        .nav-btn {
            background: #0f172a;
            color: var(--text-dim);
            border: 1px solid var(--border);
            padding: 8px 18px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .nav-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }

        /* General UI Elements */
        .btn {
            background: #334155;
            color: var(--text);
            border: 1px solid #475569;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s;
        }
        .btn:hover { background: var(--accent); border-color: var(--accent); }
        .btn-primary { background: var(--accent); border-color: var(--accent); }
        .btn-primary:hover { background: var(--accent-hover); }
        .btn-chip { background: #1e293b; border: 1px solid var(--border); color: #cbd5e1; padding: 4px 10px; border-radius: 4px; font-size: 12px; cursor: pointer; }
        .btn-chip:hover { background: var(--accent); color: #fff; }

        /* TAB 1: BLANK SCANNER STYLES */
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        .blank-layout { display: grid; grid-template-columns: 1fr 480px; gap: 20px; }
        .blank-viewer {
            background: #020617;
            border: 1px solid var(--border);
            border-radius: 12px;
            height: 840px;
            overflow: auto;
            position: relative;
            text-align: center;
            padding: 12px;
        }
        .blank-viewer img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            image-rendering: -webkit-optimize-contrast;
        }
        .blank-sidebar {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            height: 840px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .summary-card {
            background: #0f172a;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
        }
        .summary-card h3 { font-size: 15px; margin-bottom: 8px; color: var(--accent); }
        .stat-badge {
            display: inline-block;
            background: #1e293b;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            margin-right: 6px;
            margin-bottom: 6px;
            border: 1px solid var(--border);
        }

        .answers-list { display: flex; flex-direction: column; gap: 10px; }
        .ans-item {
            background: #0f172a;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
        }
        .ans-header { display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; color: var(--text-dim); margin-bottom: 6px; }
        .ans-word { font-size: 20px; font-weight: 700; letter-spacing: 2px; color: #fff; margin-bottom: 6px; }
        .chips-row { display: flex; flex-wrap: wrap; gap: 4px; }
        .chip-letter {
            background: #1e293b;
            border: 1px solid var(--border);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .chip-letter.green { border-color: var(--success); color: var(--success); }
        .chip-letter.yellow { border-color: var(--warn); color: var(--warn); }
        .chip-letter.red { border-color: var(--danger); color: var(--danger); }

        /* TAB 2: SINGLE CROP INSPECTOR */
        .inspector-layout { display: grid; grid-template-columns: 1fr 420px; gap: 20px; }
        .canvas-container {
            height: 540px;
            background: #020617;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px dashed var(--border);
        }
        .toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; align-items: center; }
        .stages-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 18px; }
        .stage-card { background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 8px; text-align: center; }
        .stage-card img { width: 100%; height: 90px; object-fit: contain; background: #1e293b; border-radius: 4px; image-rendering: pixelated; }
        .stage-card .lbl { font-size: 11px; font-weight: 600; color: var(--text-dim); margin-top: 6px; }
        .stage-card.highlight { border-color: var(--accent); }
        .stage-card.highlight .lbl { color: var(--accent); }
        .pred-item { display: flex; align-items: center; gap: 14px; background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }
        .pred-item.top { border-color: var(--success); background: rgba(34, 197, 94, 0.08); }
        .pred-glyph { font-size: 32px; font-weight: 700; width: 44px; text-align: center; }
        .pred-item.top .pred-glyph { color: var(--success); }
        .pred-body { flex: 1; }
        .pred-header { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
        .bar-wrap { background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden; }
        .bar-val { height: 100%; background: var(--accent); border-radius: 4px; }
        .pred-item.top .bar-val { background: var(--success); }

        .spinner {
            display: none;
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(15, 23, 42, 0.9);
            padding: 24px 36px;
            border-radius: 12px;
            border: 1px solid var(--accent);
            text-align: center;
            z-index: 100;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>«Дәресханә» Tatar OCR Test Suite</h1>
                <div style="font-size: 12px; color: var(--text-dim);">Full Blank Auto-Scanner & Interactive Single Character Inspector</div>
            </div>
            <div class="nav-tabs">
                <button class="nav-btn active" onclick="switchTab('blank')">📄 Full Blank Auto-Scanner</button>
                <button class="nav-btn" onclick="switchTab('cropper')">🔍 Single Crop Adjuster</button>
            </div>
        </header>

        <!-- ============================================================ -->
        <!-- TAB 1: FULL BLANK AUTO-SCANNER & GRADER                       -->
        <!-- ============================================================ -->
        <div id="tab-blank" class="tab-content active">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="display: flex; gap: 8px; align-items: center;">
                    <input type="file" id="blankFileInput" accept="image/*" style="display:none">
                    <button class="btn btn-primary" onclick="document.getElementById('blankFileInput').click()">📁 Upload Test Blank Photo</button>
                    <span style="font-size: 12px; color: var(--text-dim);">(or paste from clipboard Ctrl+V)</span>
                </div>
                <div style="display: flex; gap: 6px;">
                    <span style="font-size: 12px; color: var(--text-dim); align-self: center;">Samples:</span>
                    <button class="btn-chip" onclick="loadBlankSample('scratch/test_blank_phone_photo.jpg')">📱 Phone Photo (Tilted)</button>
                    <button class="btn-chip" onclick="loadBlankSample('scratch/test_blank_filled.png')">✍️ Filled Scan (Flat)</button>
                    <button class="btn-chip" onclick="loadBlankSample('test_blank_sample.png')">📄 Empty Template</button>
                </div>
            </div>

            <div class="blank-layout">
                <!-- Left: Full Annotated Sheet View -->
                <div class="blank-viewer" id="blankViewer">
                    <div id="blankSpinner" class="spinner">
                        <div style="font-size: 18px; font-weight: bold; margin-bottom: 8px; color: var(--accent);">Processing Blank...</div>
                        <div style="font-size: 13px; color: var(--text-dim);">Detecting ArUco corners • Rectifying • Batch OCR</div>
                    </div>
                    <img id="annotatedBlankImg" src="" alt="Upload a blank to see annotated results" style="display:none;">
                    <div id="blankPlaceholder" style="padding: 120px 20px; color: var(--text-dim);">
                        <div style="font-size: 48px; margin-bottom: 12px;">📄</div>
                        <div style="font-size: 16px; font-weight: 600;">Upload an Exam Blank to run the full pipeline</div>
                        <div style="font-size: 13px; margin-top: 6px;">Auto-rectifies perspective tilt via 4 ArUco markers, extracts every question cell, and annotates answers on the sheet.</div>
                    </div>
                </div>

                <!-- Right: Structured Answers Breakdown -->
                <div class="blank-sidebar">
                    <div class="summary-card">
                        <h3>Detection Summary</h3>
                        <div id="blankSummaryStats">
                            <div style="color: var(--text-dim); font-size: 13px;">No blank scanned yet.</div>
                        </div>
                    </div>

                    <div style="font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-dim);">
                        Recognized Answers
                    </div>

                    <div class="answers-list" id="answersList">
                        <!-- Populated by JS -->
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================ -->
        <!-- TAB 2: INTERACTIVE SINGLE CROP ADJUSTER                       -->
        <!-- ============================================================ -->
        <div id="tab-cropper" class="tab-content">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="font-size: 13px; color: var(--text-dim);">Tightly frame any character crop. Live 64x64 CNN input and Top-5 candidates update in real time.</div>
                <input type="file" id="cropFileInput" accept="image/*" style="display:none">
                <button class="btn btn-primary" onclick="document.getElementById('cropFileInput').click()">Upload Image</button>
            </div>

            <div class="inspector-layout">
                <div class="card">
                    <div class="canvas-container">
                        <img id="cropperImage" src="/sample/strip2_rot90_cw.png" alt="Crop Target">
                    </div>

                    <div class="toolbar">
                        <button class="btn" onclick="cropper.rotate(-90)">↺ 90°</button>
                        <button class="btn" onclick="cropper.rotate(90)">↻ 90°</button>
                        <button class="btn" onclick="cropper.zoom(0.15)">+</button>
                        <button class="btn" onclick="cropper.zoom(-0.15)">−</button>
                        <button class="btn" onclick="cropper.reset()">Reset</button>

                        <div style="margin-left: auto; display: flex; gap: 8px; align-items: center;">
                            <label style="font-size: 12px; color: var(--text-dim);">Contrast:</label>
                            <select id="contrastSelect" onchange="triggerCropInference()" style="background:#1e293b; color:#fff; border:1px solid var(--border); font-size:12px; padding:4px 8px; border-radius:4px;">
                                <option value="percentile" selected>Percentile Min/Max</option>
                                <option value="clahe">Adaptive CLAHE</option>
                            </select>
                        </div>
                    </div>

                    <div style="display: flex; gap: 6px; margin-top: 10px; align-items: center;">
                        <span style="font-size: 12px; color: var(--text-dim);">Quick Strips:</span>
                        <button class="btn-chip" onclick="loadCropUrl('/sample/strip2_rot90_cw.png')">Full 6-Cell Strip</button>
                        <button class="btn-chip" onclick="loadCropUrl('/sample/crop_raw_1.png')">Ә (Cell 1)</button>
                        <button class="btn-chip" onclick="loadCropUrl('/sample/crop_raw_4.png')">Җ (Cell 4)</button>
                        <button class="btn-chip" onclick="loadCropUrl('/sample/crop_raw_5.png')">Щ (Cell 5)</button>
                        <button class="btn-chip" onclick="loadCropUrl('/sample/crop_raw_6.png')">Ч (Cell 6)</button>
                    </div>
                </div>

                <div class="card">
                    <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; color: var(--text-dim); margin-bottom: 8px;">Model Tensor Input</div>
                    <div class="stages-row">
                        <div class="stage-card">
                            <img id="imgCropped" src="" alt="Cropped">
                            <div class="lbl">1. User Crop</div>
                        </div>
                        <div class="stage-card">
                            <img id="imgPadded" src="" alt="Padded">
                            <div class="lbl">2. Square Pad</div>
                        </div>
                        <div class="stage-card highlight">
                            <img id="imgNorm64" src="" alt="64x64">
                            <div class="lbl">3. CNN 64x64</div>
                        </div>
                    </div>

                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 13px; font-weight: 600; text-transform: uppercase; color: var(--text-dim);">Top-5 Model Predictions</span>
                        <span id="cropLatency" style="font-size: 11px; color: var(--success);">Live</span>
                    </div>

                    <div id="cropPredictionsList">
                        <!-- Populated by JS -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentTab = 'blank';
        let cropper = null;
        let debounceTimer = null;

        function switchTab(tab) {
            currentTab = tab;
            document.querySelectorAll('.nav-btn').forEach((b, i) => b.classList.toggle('active', (i === 0 && tab === 'blank') || (i === 1 && tab === 'cropper')));
            document.getElementById('tab-blank').classList.toggle('active', tab === 'blank');
            document.getElementById('tab-cropper').classList.toggle('active', tab === 'cropper');
            if (tab === 'cropper' && !cropper) initCropper();
        }

        // --- TAB 1: BLANK SCANNER LOGIC ---
        const blankFileInput = document.getElementById('blankFileInput');
        blankFileInput.addEventListener('change', (e) => {
            if (e.target.files.length) uploadAndScanBlank(e.target.files[0]);
        });

        // Paste support
        window.addEventListener('paste', (e) => {
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            for (let item of items) {
                if (item.type.indexOf('image') !== -1) {
                    const blob = item.getAsFile();
                    if (currentTab === 'blank') uploadAndScanBlank(blob);
                    else loadCropBlob(blob);
                    break;
                }
            }
        });

        function loadBlankSample(path) {
            fetch('/sample/' + path)
                .then(r => r.blob())
                .then(blob => uploadAndScanBlank(blob));
        }

        function uploadAndScanBlank(fileOrBlob) {
            const reader = new FileReader();
            reader.onload = function(evt) {
                const b64 = evt.target.result.split(',')[1];
                document.getElementById('blankSpinner').style.display = 'block';

                fetch('/api/scan_blank', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_b64: b64 })
                })
                .then(r => r.json())
                .then(data => {
                    document.getElementById('blankSpinner').style.display = 'none';
                    if (data.error) {
                        alert('Error scanning blank: ' + data.error);
                        return;
                    }
                    renderBlankResults(data);
                })
                .catch(err => {
                    document.getElementById('blankSpinner').style.display = 'none';
                    alert('Network error scanning blank');
                });
            };
            reader.readAsDataURL(fileOrBlob);
        }

        function renderBlankResults(data) {
            document.getElementById('blankPlaceholder').style.display = 'none';
            const img = document.getElementById('annotatedBlankImg');
            img.src = data.annotated_b64;
            img.style.display = 'block';

            // Summary Stats
            const stats = data.stats;
            document.getElementById('blankSummaryStats').innerHTML = `
                <div><span class="stat-badge">Alignment: <strong>${data.rectification_method}</strong></span></div>
                <div>
                    <span class="stat-badge">Recognized Letters: <strong>${stats.total_letters}</strong></span>
                    <span class="stat-badge" style="color:var(--success)">Avg Confidence: <strong>${stats.avg_confidence}%</strong></span>
                </div>
                <div style="margin-top: 8px; font-size: 13px;">
                    Student: <strong style="color:#fff; font-size:15px;">${data.student_name || '—'}</strong>
                </div>
            `;

            // Answers List
            const list = document.getElementById('answersList');
            list.innerHTML = '';

            // 1. Student Name Card
            const nameCard = document.createElement('div');
            nameCard.className = 'ans-item';
            nameCard.innerHTML = `
                <div class="ans-header"><span>STUDENT IDENTIFICATION</span><span>16 CELLS</span></div>
                <div class="ans-word">${data.student_name || '(Empty)'}</div>
            `;
            list.appendChild(nameCard);

            // 2. Questions Cards
            data.questions.forEach(q => {
                const item = document.createElement('div');
                item.className = 'ans-item';
                
                let chipsHtml = '';
                q.cells.forEach(c => {
                    if (!c.is_empty) {
                        const colClass = c.confidence >= 80 ? 'green' : (c.confidence >= 50 ? 'yellow' : 'red');
                        chipsHtml += `<span class="chip-letter ${colClass}" title="Conf: ${c.confidence}%">${c.char}</span>`;
                    }
                });

                item.innerHTML = `
                    <div class="ans-header">
                        <span>QUESTION №${q.q_num}</span>
                        <span>${q.cells.filter(c => !c.is_empty).length} LETTERS</span>
                    </div>
                    <div class="ans-word">${q.text || '—'}</div>
                    <div class="chips-row">${chipsHtml || '<span style="color:var(--text-dim); font-size:12px;">No answer written</span>'}</div>
                `;
                list.appendChild(item);
            });
        }

        // --- TAB 2: CROP INSPECTOR LOGIC ---
        const cropImg = document.getElementById('cropperImage');
        const cropFileInput = document.getElementById('cropFileInput');
        cropFileInput.addEventListener('change', (e) => {
            if (e.target.files.length) loadCropBlob(e.target.files[0]);
        });

        function initCropper() {
            if (cropper) cropper.destroy();
            cropper = new Cropper(cropImg, {
                viewMode: 1,
                dragMode: 'move',
                autoCrop: true,
                autoCropArea: 0.35,
                cropBoxMovable: true,
                cropBoxResizable: true,
                ready() { triggerCropInference(); },
                crop() {
                    clearTimeout(debounceTimer);
                    debounceTimer = setTimeout(triggerCropInference, 120);
                }
            });
        }

        function loadCropBlob(blob) {
            const reader = new FileReader();
            reader.onload = (e) => { cropImg.src = e.target.result; initCropper(); };
            reader.readAsDataURL(blob);
        }

        function loadCropUrl(url) {
            cropImg.src = url;
            initCropper();
        }

        function triggerCropInference() {
            if (!cropper) return;
            const canvas = cropper.getCroppedCanvas({ imageSmoothingEnabled: true, imageSmoothingQuality: 'high' });
            if (!canvas) return;

            const b64 = canvas.toDataURL('image/png').split(',')[1];
            const mode = document.getElementById('contrastSelect').value;
            const t0 = performance.now();

            fetch('/api/predict_box', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_b64: b64, contrast_mode: mode })
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('cropLatency').innerText = (performance.now() - t0).toFixed(0) + ' ms';
                document.getElementById('imgCropped').src = data.stages.cropped;
                document.getElementById('imgPadded').src = data.stages.padded;
                document.getElementById('imgNorm64').src = data.stages.normalized_64x64;

                const list = document.getElementById('cropPredictionsList');
                list.innerHTML = '';
                data.predictions_39.forEach((p, idx) => {
                    const isTop = idx === 0;
                    const item = document.createElement('div');
                    item.className = 'pred-item ' + (isTop ? 'top' : '');
                    item.innerHTML = `
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
                    list.appendChild(item);
                });
            });
        }

        // Initial sample auto-load
        window.addEventListener('DOMContentLoaded', () => {
            loadBlankSample('scratch/test_blank_phone_photo.jpg');
        });
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
            rel_path = self.path[len("/sample/"):]
            sample_path = WORKSPACE_DIR / rel_path
            if not sample_path.exists():
                sample_path = WORKSPACE_DIR / "dataset" / "six_cells_test" / os.path.basename(rel_path)
            if sample_path.exists():
                self.send_response(200)
                ext = sample_path.suffix.lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                self.send_header("Content-Type", mime)
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
        if self.path == "/api/scan_blank":
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len)
            try:
                data = json.loads(post_body.decode('utf-8'))
                img_bytes = base64.b64decode(data['image_b64'])
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                res = blank_scanner.process_blank(img_bgr, annotate=True)

                # Encode annotated image to base64
                _, buf = cv2.imencode(".jpg", cv2.resize(res["annotated_bgr"], (1050, 1485)), [cv2.IMWRITE_JPEG_QUALITY, 85])
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")

                response_payload = {
                    "status": "success",
                    "rectification_method": res["rectification_method"],
                    "student_name": res["student_name"],
                    "questions": res["questions"],
                    "stats": res["stats"],
                    "annotated_b64": annotated_b64
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        elif self.path == "/api/predict_box":
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len)
            try:
                data = json.loads(post_body.decode('utf-8'))
                img_bytes = base64.b64decode(data['image_b64'])
                contrast_mode = data.get('contrast_mode', 'percentile')

                res = process_cropped_area(img_bytes, contrast_mode=contrast_mode)
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
    print(f"\n==================================================================")
    print(f"  «Дәресханә» Tatar OCR Suite is RUNNING!")
    print(f"  Open in browser: http://127.0.0.1:{PORT}")
    print(f"==================================================================\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
