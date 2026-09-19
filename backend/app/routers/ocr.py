import base64
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from app.schemas.ocr import (
    ScanBlankRequest,
    ScanBlankResponse,
    PredictBoxRequest,
    PredictBoxResponse,
)
from app.services.ocr_service import OCRService

router = APIRouter(prefix="/ocr", tags=["Blank OCR & Recognition"])
legacy_router = APIRouter(tags=["Blank OCR (Direct / Compatibility)"])


@router.post(
    "/scan-blank",
    response_model=ScanBlankResponse,
    status_code=status.HTTP_200_OK,
    summary="Scan & Recognize Tatar Test Blank (Base64)",
)
async def scan_blank_endpoint(payload: ScanBlankRequest):
    try:
        service = OCRService.get_instance()
        img_bgr = service.decode_image_from_b64(payload.image_b64)
        result = service.scan_blank(img_bgr, annotate=payload.annotate)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process blank: {e}")


@router.post(
    "/scan-blank-upload",
    response_model=ScanBlankResponse,
    status_code=status.HTTP_200_OK,
    summary="Scan & Recognize Tatar Test Blank (File Upload)",
)
async def scan_blank_upload_endpoint(
    file: UploadFile = File(...),
    annotate: bool = Form(True),
):
    try:
        content = await file.read()
        service = OCRService.get_instance()
        img_bgr = service.decode_image_from_bytes(content)
        result = service.scan_blank(img_bgr, annotate=annotate)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process blank upload: {e}")


@router.post(
    "/predict-box",
    response_model=PredictBoxResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Letter in Cropped Cell Box",
)
async def predict_box_endpoint(payload: PredictBoxRequest):
    try:
        service = OCRService.get_instance()
        raw_b64 = payload.image_b64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        result = service.predict_box(img_bytes, contrast_mode=payload.contrast_mode)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict cell: {e}")


# Legacy / Direct Compatibility Endpoints matching ocr_inspector_server.py
@legacy_router.post(
    "/api/scan_blank",
    response_model=ScanBlankResponse,
    status_code=status.HTTP_200_OK,
    summary="Scan Blank (Legacy route /api/scan_blank)",
    include_in_schema=True,
)
async def legacy_scan_blank(payload: ScanBlankRequest):
    return await scan_blank_endpoint(payload)


@legacy_router.post(
    "/api/predict_box",
    response_model=PredictBoxResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Box (Legacy route /api/predict_box)",
    include_in_schema=True,
)
async def legacy_predict_box(payload: PredictBoxRequest):
    return await predict_box_endpoint(payload)
