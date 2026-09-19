import sys
import base64
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(r"c:\Users\galee\PycharmProjects\tatar_ocr")
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app

def run_verification():
    print("Testing OCR API in Main FastAPI Backend...")
    client = TestClient(app)

    # 1. Health check
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    print("  ✓ Health check passed")

    # 2. Test blank image
    sample_img_path = PROJECT_ROOT / "scratch" / "test_blank_phone_photo.jpg"
    if not sample_img_path.exists():
        sample_img_path = PROJECT_ROOT / "test_blank_sample.png"

    assert sample_img_path.exists(), f"Sample image not found: {sample_img_path}"

    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("ascii")

    # 3. Test POST /api/v1/ocr/scan-blank (Base64)
    print("  Testing POST /api/v1/ocr/scan-blank ...")
    resp = client.post(
        "/api/v1/ocr/scan-blank",
        json={"image_b64": img_b64, "annotate": True}
    )
    assert resp.status_code == 200, f"/api/v1/ocr/scan-blank failed ({resp.status_code}): {resp.text}"
    data = resp.json()
    assert data.get("status") == "success", f"Unexpected status: {data}"
    assert "rectification_method" in data
    assert "student_name" in data
    assert "questions" in data
    assert "stats" in data
    assert data.get("annotated_b64") is not None
    print(f"  ✓ /api/v1/ocr/scan-blank success! Method: {data['rectification_method']}, Student: '{data['student_name']}', Letters: {data['stats']['total_letters']}")

    # 4. Test POST /api/scan_blank (Legacy route)
    print("  Testing POST /api/scan_blank ...")
    resp_legacy = client.post(
        "/api/scan_blank",
        json={"image_b64": img_b64, "annotate": False}
    )
    assert resp_legacy.status_code == 200, f"/api/scan_blank failed ({resp_legacy.status_code}): {resp_legacy.text}"
    data_legacy = resp_legacy.json()
    assert data_legacy.get("status") == "success"
    print("  ✓ /api/scan_blank legacy endpoint passed")

    # 5. Test POST /api/v1/ocr/scan-blank-upload (Multipart upload)
    print("  Testing POST /api/v1/ocr/scan-blank-upload ...")
    with open(sample_img_path, "rb") as f:
        resp_upload = client.post(
            "/api/v1/ocr/scan-blank-upload",
            files={"file": ("blank.jpg", f, "image/jpeg")},
            data={"annotate": "true"}
        )
    assert resp_upload.status_code == 200, f"Upload failed ({resp_upload.status_code}): {resp_upload.text}"
    data_upload = resp_upload.json()
    assert data_upload.get("status") == "success"
    print("  ✓ /api/v1/ocr/scan-blank-upload multipart upload passed")

    # 6. Test POST /api/v1/ocr/predict-box
    print("  Testing POST /api/v1/ocr/predict-box ...")
    # Take a small crop or simple dummy 64x64 image
    import cv2
    import numpy as np
    dummy_cell = np.full((64, 64, 3), 245, dtype=np.uint8)
    cv2.putText(dummy_cell, "А", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 2)
    _, buf = cv2.imencode(".png", dummy_cell)
    cell_b64 = base64.b64encode(buf).decode("ascii")

    resp_box = client.post(
        "/api/v1/ocr/predict-box",
        json={"image_b64": cell_b64, "contrast_mode": "percentile"}
    )
    assert resp_box.status_code == 200, f"Predict box failed ({resp_box.status_code}): {resp_box.text}"
    data_box = resp_box.json()
    assert "predictions_39" in data_box
    top_pred = data_box["predictions_39"][0]
    print(f"  ✓ /api/v1/ocr/predict-box passed! Top prediction: '{top_pred['char']}' ({top_pred['confidence']}%)")

    print("\nALL OCR API TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_verification()
