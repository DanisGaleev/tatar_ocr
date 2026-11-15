"""
Tatar OCR - Pipeline Visualizer & Web Uploader.
Простой инструмент для загрузки и визуализации распознавания любых бланков:
- Поддерживает любой бланк (JPG, PNG) без метаданных.
- Работает в CLI (консольный вывод + аннотированное изображение)
  и в Web-режиме (drag-and-drop интерфейс в браузере).
- Использует актуальную модель v4 INT8.
"""

import sys
import json
import time
import base64
import argparse
import webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import cv2
import numpy as np

from blank_pipeline import BlankOCRScanner

DEFAULT_MODEL = "models/tatar_ocr_v4_handwritten39_int8.onnx"
LOW_CONF_THRESHOLD = 85.0
ANNOTATED_MAX_WIDTH = 1400
JPEG_QUALITY = 82
DEFAULT_PORT = 8080
PORT_SCAN_RANGE = 10


def _serialize_cells(cells: list[dict]) -> list[dict]:
    """Extract JSON-safe cell prediction dicts including Top-3 candidates."""
    serialized = []
    for cell in cells:
        raw_top3 = cell.get("top3") or cell.get("result", {}).get("top3", [])
        top3 = [(str(ch), round(float(conf), 1)) for ch, conf in raw_top3]
        cell_idx = cell.get("cell_idx")
        serialized.append({
            "cell_idx": int(cell_idx) if cell_idx is not None else None,
            "char": cell.get("char", " "),
            "confidence": round(float(cell.get("confidence", 0.0)), 1),
            "is_empty": bool(cell.get("is_empty", True)),
            "top3": top3,
        })
    return serialized


def _build_clean_result(result: dict) -> dict:
    """Build a JSON-serializable summary payload from a BlankOCRScanner result."""
    return {
        "status": result.get("status"),
        "student_name": result.get("student_name"),
        "student_name_cells": _serialize_cells(result.get("student_name_cells", [])),
        "rectification_method": result.get("rectification_method"),
        "questions": [
            {
                "q_num": q.get("q_num"),
                "text": q.get("text"),
                "cells": _serialize_cells(q.get("cells", [])),
            }
            for q in result.get("questions", [])
        ],
        "stats": result.get("stats", {}),
    }


def format_cli_output(result: dict) -> str:
    """Форматирует результат распознавания для вывода в терминал."""
    lines = []
    lines.append("=" * 65)
    lines.append("РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ БЛАНКА")
    lines.append("=" * 65)
    lines.append(f"Метод выравнивания: {result.get('rectification_method', 'N/A')}")
    lines.append(f"ФИО ученика:        {result.get('student_name') or '(не заполнено)'}")
    lines.append("-" * 65)
    lines.append("ОТВЕТЫ НА ВОПРОСЫ:")
    
    questions = result.get("questions", [])
    for q in questions:
        q_num = q.get("q_num")
        text = q.get("text", "")
        cells = q.get("cells", [])
        
        # Средняя уверенность по заполненным буквам
        filled_confs = [c.get("confidence", 0.0) for c in cells if not c.get("is_empty")]
        avg_conf = (sum(filled_confs) / len(filled_confs)) if filled_confs else 100.0
        
        # Подсветка сомнительных букв (< LOW_CONF_THRESHOLD)
        uncertain = [
            f"яч.{c['cell_idx']} '{c['char']}' ({c['confidence']:.1f}%)"
            for c in cells
            if not c.get("is_empty") and c.get("confidence", 0.0) < LOW_CONF_THRESHOLD
        ]
        unc_str = f"  [Внимание: {', '.join(uncertain)}]" if uncertain else "  [OK]"
        
        lines.append(f"  Вопрос {q_num:>2d}: {text:<16s} (уверенность: {avg_conf:5.1f}%){unc_str}")

    stats = result.get("stats", {})
    lines.append("-" * 65)
    lines.append(f"Всего распознано букв: {stats.get('total_letters', 0)}")
    lines.append(f"Средняя уверенность:   {stats.get('avg_confidence', 0.0):.1f}%")
    lines.append("=" * 65)
    return "\n".join(lines)


def process_single_image(scanner: BlankOCRScanner, image_path: Path, output_dir: Path = None):
    """Обрабатывает одно изображение, печатает результат и сохраняет аннотацию."""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[!] Файл не найден: {image_path}")
        return None

    t0 = time.perf_counter()
    result = scanner.process_image(image_path, annotate=True)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print(format_cli_output(result))
    print(f"Время обработки: {dt_ms:.1f} мс")

    # Сохранение аннотированного изображения
    if output_dir is None:
        output_dir = WORKSPACE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_img_path = output_dir / f"annotated_{image_path.name}"
    if result.get("annotated_bgr") is not None:
        cv2.imwrite(str(out_img_path), result["annotated_bgr"])
        print(f"[✓] Аннотированное изображение сохранено: {out_img_path}")

    # Сохранение JSON
    out_json_path = output_dir / f"result_{image_path.stem}.json"
    clean_res = _build_clean_result(result)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(clean_res, f, ensure_ascii=False, indent=2)
    print(f"[✓] JSON результат сохранен: {out_json_path}")
    return result


# ---------------------------------------------------------------------------
# Встроенный легковесный Web UI (Drag-and-Drop Uploader)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tatar OCR • Распознавание бланков</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background-color: #0f172a; color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; }
    .drop-zone { border: 2px dashed #3b82f6; transition: all 0.2s; }
    .drop-zone.dragover { border-color: #10b981; background-color: rgba(16, 185, 129, 0.08); }
  </style>
</head>
<body class="min-h-screen p-4 md:p-8">
  <div class="max-w-6xl mx-auto space-y-6">
    
    <!-- Заголовок -->
    <header class="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-slate-800 gap-4">
      <div>
        <h1 class="text-2xl md:text-3xl font-extrabold tracking-tight text-white flex items-center gap-3">
          <span class="p-2 bg-blue-600 rounded-lg text-white">📝</span>
          Tatar OCR • Распознавание бланков
        </h1>
        <p class="text-slate-400 text-sm mt-1">
          Загрузите фотографию любого бланка с камеры смартфона или сканера (модель v4 INT8)
        </p>
      </div>
      <div class="flex items-center gap-2">
        <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
          ● v4 INT8 MultiScale (1.0 MB)
        </span>
      </div>
    </header>

    <!-- Зона загрузки -->
    <div id="dropZone" class="drop-zone rounded-2xl p-8 text-center cursor-pointer bg-slate-900/60 hover:bg-slate-900">
      <input type="file" id="fileInput" accept="image/*" class="hidden">
      <div class="flex flex-col items-center justify-center space-y-3">
        <div class="p-4 bg-blue-500/10 rounded-full text-blue-400 text-3xl">📤</div>
        <div class="text-lg font-medium text-slate-200">
          Перетащите сюда фото бланка или <span class="text-blue-400 underline">выберите файл</span>
        </div>
        <p class="text-xs text-slate-500">Поддерживаются форматы JPG, PNG, WEBP любого разрешения</p>
      </div>
    </div>

    <!-- Индикатор загрузки -->
    <div id="loading" class="hidden text-center py-12">
      <div class="inline-block animate-spin rounded-full h-10 w-10 border-4 border-blue-500 border-t-transparent"></div>
      <p class="text-slate-400 mt-3 text-sm">Выравнивание перспективы и распознавание символов...</p>
    </div>

    <!-- Результаты -->
    <div id="results" class="hidden space-y-6">
      
      <!-- Сводная плашка -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl">
          <div class="text-xs text-slate-400 uppercase font-semibold">ФИО Ученика</div>
          <div id="resName" class="text-xl font-bold text-white mt-1"></div>
          <div id="resNameCells" class="flex flex-wrap gap-1 mt-2"></div>
        </div>
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl">
          <div class="text-xs text-slate-400 uppercase font-semibold">Всего букв</div>
          <div id="resLetters" class="text-xl font-bold text-white mt-1"></div>
        </div>
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl">
          <div class="text-xs text-slate-400 uppercase font-semibold">Средняя уверенность</div>
          <div id="resConf" class="text-xl font-bold text-emerald-400 mt-1"></div>
        </div>
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl">
          <div class="text-xs text-slate-400 uppercase font-semibold">Выравнивание листа</div>
          <div id="resMethod" class="text-sm font-medium text-slate-300 mt-2 truncate"></div>
        </div>
      </div>

      <!-- Главный блок: Ответы + Аннотированное фото -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        <!-- Колонка с ответами -->
        <div class="lg:col-span-5 space-y-3">
          <h2 class="text-lg font-bold text-slate-200">Ответы на вопросы</h2>
          <div id="questionsList" class="space-y-3"></div>
        </div>

        <!-- Колонка с аннотированным изображением -->
        <div class="lg:col-span-7 space-y-3">
          <h2 class="text-lg font-bold text-slate-200">Аннотированный бланк</h2>
          <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden p-2">
            <img id="annotatedImage" class="w-full h-auto rounded-lg" alt="Распознанный бланк">
          </div>
        </div>

      </div>

    </div>

  </div>

  <script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
      }
    });

    function handleFile(file) {
      if (!file.type.startsWith('image/')) {
        alert('Пожалуйста, выберите файл изображения (JPG, PNG)');
        return;
      }
      loading.classList.remove('hidden');
      results.classList.add('hidden');

      const formData = new FormData();
      formData.append('file', file);

      fetch('/api/recognize', {
        method: 'POST',
        body: formData
      })
      .then(res => res.json())
      .then(data => {
        loading.classList.add('hidden');
        if (data.status === 'success') {
          displayResults(data);
        } else {
          alert('Ошибка распознавания: ' + (data.error || 'Неизвестная ошибка'));
        }
      })
      .catch(err => {
        loading.classList.add('hidden');
        alert('Сетевая ошибка при загрузке: ' + err);
      });
    }

    function renderCellBadge(c, showEmpty) {
      if (c.is_empty) {
        return showEmpty ? '<span class="inline-block w-8 h-10 border border-slate-800 rounded bg-slate-950/40"></span>' : '';
      }
      const borderCol = c.confidence >= 85
        ? 'border-emerald-500/40 text-emerald-300'
        : 'border-amber-500/60 text-amber-300 bg-amber-500/10';
      const top3Tooltip = (c.top3 || []).map(t => `${t[0]}: ${t[1]}%`).join(' | ');
      return `<span title="Top-3: ${top3Tooltip}" class="inline-flex flex-col items-center justify-center w-8 h-10 border ${borderCol} rounded font-bold text-base bg-slate-800/80 cursor-help">${c.char}</span>`;
    }

    function displayResults(data) {
      document.getElementById('resName').textContent = data.student_name || '(не заполнено)';
      document.getElementById('resNameCells').innerHTML = (data.student_name_cells || []).map(c => renderCellBadge(c, false)).join('');
      document.getElementById('resLetters').textContent = data.stats.total_letters;
      document.getElementById('resConf').textContent = data.stats.avg_confidence + '%';
      document.getElementById('resMethod').textContent = data.rectification_method;

      if (data.annotated_b64) {
        document.getElementById('annotatedImage').src = data.annotated_b64;
      }

      const qList = document.getElementById('questionsList');
      qList.innerHTML = '';

      data.questions.forEach(q => {
        const card = document.createElement('div');
        card.className = 'bg-slate-900 border border-slate-800 rounded-xl p-4';

        let badgeCol = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
        let filledConfs = q.cells.filter(c => !c.is_empty).map(c => c.confidence);
        let qAvg = filledConfs.length ? (filledConfs.reduce((a,b)=>a+b,0) / filledConfs.length).toFixed(1) : 100;
        if (qAvg < 85) badgeCol = 'bg-amber-500/10 text-amber-400 border-amber-500/30';
        if (qAvg < 70) badgeCol = 'bg-rose-500/10 text-rose-400 border-rose-500/30';

        const cellsHtml = q.cells.map(c => renderCellBadge(c, true)).join('');

        card.innerHTML = `
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-semibold uppercase text-slate-400">Вопрос #${q.q_num}</span>
            <span class="px-2 py-0.5 rounded text-xs font-bold border ${badgeCol}">${qAvg}%</span>
          </div>
          <div class="text-xl font-bold tracking-wider text-white mb-3">${q.text || '<span class="text-slate-600 font-normal text-sm">пусто</span>'}</div>
          <div class="flex flex-wrap gap-1.5">${cellsHtml}</div>
        `;
        qList.appendChild(card);
      });

      results.classList.remove('hidden');
    }
  </script>
</body>
</html>
"""


class OCRRequestHandler(BaseHTTPRequestHandler):
    scanner = None

    def log_message(self, format, *args):
        # Тихий лог для чистоты консоли
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/recognize":
            ctype = self.headers.get("Content-Type", "")
            if not ctype.startswith("multipart/form-data"):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"status": "error", "error": "Invalid Content-Type"}')
                return

            # Чтение multipart тела
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Простой парсинг бинарного изображения из multipart
            boundary = ctype.split("boundary=")[-1].encode("utf-8")
            parts = body.split(boundary)
            img_bytes = None
            for p in parts:
                if b'filename="' in p and (b'\r\n\r\n' in p):
                    header, data = p.split(b'\r\n\r\n', 1)
                    # Срезка завершающего \r\n--
                    img_bytes = data.rstrip(b'\r\n--')
                    break

            if not img_bytes:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"status": "error", "error": "No image data found"}')
                return

            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"status": "error", "error": "Cannot decode image"}')
                return

            # Запуск распознавания
            res = self.scanner.process_blank(img_bgr, annotate=True)

            # Конвертация аннотированного скана в Base64
            ann_b64 = ""
            if res.get("annotated_bgr") is not None:
                ann = res["annotated_bgr"]
                h, w = ann.shape[:2]
                if w > ANNOTATED_MAX_WIDTH:
                    scale = ANNOTATED_MAX_WIDTH / float(w)
                    ann = cv2.resize(ann, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                succ, buf = cv2.imencode(".jpg", ann, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if succ:
                    ann_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

            response_data = _build_clean_result(res)
            response_data["status"] = "success"
            response_data["annotated_b64"] = ann_b64

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_web_server(scanner: BlankOCRScanner, port: int = DEFAULT_PORT):
    """Запускает простой локальный веб-сервер для загрузки фото бланков."""
    OCRRequestHandler.scanner = scanner

    # Попытка открыть порт (если занят, берем следующий)
    httpd = None
    for p in range(port, port + PORT_SCAN_RANGE):
        try:
            httpd = HTTPServer(("127.0.0.1", p), OCRRequestHandler)
            port = p
            break
        except OSError:
            continue

    if httpd is None:
        print("[!] Не удалось открыть порт для веб-сервера.")
        return

    url = f"http://127.0.0.1:{port}"
    print("=" * 65)
    print(f"[*] Веб-интерфейс Tatar OCR запущен на: {url}")
    print("[*] Перетащите фото любого бланка в браузере для распознавания.")
    print("[*] Нажмите Ctrl+C в терминале для остановки.")
    print("=" * 65)

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Веб-сервер остановлен.")
        httpd.server_close()


def main():
    """CLI entry point for single-image OCR visualization or Web UI server."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Tatar OCR - Простой просмотрщик и распознаватель бланков")
    parser.add_argument("image", nargs="?", default=None, help="Путь к фото любого бланка (JPG, PNG). Если не указан — открывается веб-интерфейс загрузки.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Путь к модели (по умолчанию: {DEFAULT_MODEL})")
    parser.add_argument("--output", default="output", help="Директория для сохранения результатов")
    parser.add_argument("--web", action="store_true", help="Принудительно запустить локальный веб-интерфейс для загрузки файлов")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Порт локального веб-сервера (по умолчанию: {DEFAULT_PORT})")
    args = parser.parse_args()

    # Инициализация сканера
    print(f"[*] Инициализация Tatar OCR Scanner (модель: {args.model})...")
    scanner = BlankOCRScanner(model_path=args.model)

    if args.web or args.image is None:
        # Режим 1: Веб-интерфейс загрузки
        run_web_server(scanner, port=args.port)
    else:
        # Режим 2: Консольная обработка переданного изображения
        img_path = Path(args.image)
        process_single_image(scanner, img_path, output_dir=Path(args.output))


if __name__ == "__main__":
    main()
