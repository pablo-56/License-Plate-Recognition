# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any
from PIL import Image
import numpy as np
import pytesseract
from pytesseract import Output
import re
import os

# OPTIONAL: set tesseract path explicitly if needed
# pytesseract.pytesseract.tesseract_cmd = r"/usr/share/tesseract-ocr/5/tessdata"
#/usr/share/tesseract-ocr/5/tessdata 

# -------------------------
#  Project / model settings
# -------------------------

# Absolute project root for your LPR project
PROJECT_ROOT = Path("/home/pablo/Desktop/lpd")

# Absolute path to your trained YOLO weights
YOLO_WEIGHTS_PATH = PROJECT_ROOT / "notebooks"  / "runs" / "detect" / "lpr-car-plate-yolov8n-cpu" / "weights" / "best.pt"
if not YOLO_WEIGHTS_PATH.exists():
    raise RuntimeError(f"YOLO weights not found at {YOLO_WEIGHTS_PATH}")

# Explicit path to tessdata (where lpr.traineddata lives)
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/5/tessdata/"

# OCR language code for my fine-tuned Tesseract model (lpr.traineddata)
OCR_LANG = "lpr"   # uses /usr/share/tesseract-ocr/5/tessdata/lpr.traineddata

# Load YOLO model once at startup
yolo_model = YOLO(str(YOLO_WEIGHTS_PATH))

# -------------
#  FastAPI app
# -------------

app = FastAPI(
    title="LPR API",
    version="0.1.0",
    description="License Plate Recognition API: YOLOv8 + Tesseract (lpr)",
)

# Allow browser apps during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
#  Helper: run Tesseract on crop
# -----------------------------

def ocr_plate_pytesseract(img: Image.Image) -> Dict[str, Any]:
    """
    Run Tesseract OCR on a cropped plate image and return:
    {
      "text": "ABC123",
      "confidence": 87.5
    }
    """
    # Convert to grayscale (optional but often helps)
    gray = img.convert("L")

    # PSM 7: treat image as a single text line (good for license plates)
    data = pytesseract.image_to_data(
        gray,
        lang=OCR_LANG,
        config="--psm 7",
        output_type=Output.DICT,
    )

    texts = data["text"]
    confs = data["conf"]

    words = []
    word_confs = []
    for t, c in zip(texts, confs):
        try:
            c = int(c)
        except ValueError:
            c = -1
        if c < 0:
            continue
        t = t.strip()
        if not t:
            continue
        words.append(t)
        word_confs.append(c)

    if not words:
        return {"text": "", "confidence": 0.0}

    raw_text = "".join(words)
    # Keep only alphanumeric characters, uppercase (typical plate style)
    clean_text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())

    avg_conf = float(sum(word_confs) / len(word_confs)) if word_confs else 0.0

    return {"text": clean_text, "confidence": avg_conf}


# -----------------------------
#  Helper: run YOLO on an image
# -----------------------------

def detect_plates_yolo(
    img: Image.Image,
    conf_threshold: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Run YOLO detector on a PIL image and return a list of dicts:
    {
      "bbox": {"x1": int, "y1": int, "x2": int, "y2": int},
      "confidence": float
    }
    """
    w, h = img.size
    np_img = np.array(img)

    # Run YOLO prediction (first result only)
    results = yolo_model.predict(
        source=np_img,
        device="cpu",        # safe for your 4GB GPU; use 0 to force GPU
        imgsz=320,
        conf=conf_threshold,
        verbose=False,
    )[0]

    detections: List[Dict[str, Any]] = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0].item())

        # Clamp to image bounds
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))

        detections.append(
            {
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "confidence": score,
            }
        )

    return detections


# ------------------------------------------
#  API endpoint: full LPR pipeline in one go
# ------------------------------------------

@app.post("/v1/plates/detect", summary="Detect and read license plates in an image")
async def detect_and_read_plates(
    file: UploadFile = File(...),
    conf_threshold: float = 0.4,
):
    """
    Accept an image file, run YOLO plate detection + Tesseract OCR, and return detections:
    [
      {
        "bbox": {"x1": ..., "y1": ..., "x2": ..., "y2": ...},
        "detection_confidence": 0.97,
        "plate_text": "ABC123",
        "ocr_confidence": 87.5
      },
      ...
    ]
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    # Read file into memory
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # Open as PIL image
    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {e}")

    # Step 1: detect plates
    detections = detect_plates_yolo(img, conf_threshold=conf_threshold)

    results: List[Dict[str, Any]] = []

    # Step 2: for each bbox, crop + OCR
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

        # Crop plate region
        crop = img.crop((x1, y1, x2, y2))

        # OCR with Tesseract lpr model
        ocr_result = ocr_plate_pytesseract(crop)

        results.append(
            {
                "bbox": bbox,
                "detection_confidence": det["confidence"],
                "plate_text": ocr_result["text"],
                "ocr_confidence": ocr_result["confidence"],
            }
        )

    return {
        "image_width": img.size[0],
        "image_height": img.size[1],
        "num_detections": len(results),
        "detections": results,
    }
