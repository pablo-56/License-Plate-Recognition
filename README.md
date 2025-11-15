# License Plate Recognition (YOLOv8 + Tesseract)

End-to-end **License Plate Recognition (LPR)** pipeline:

- **YOLOv8** detects license plate bounding boxes
- **Tesseract OCR** (fine-tuned `lpr.traineddata`) reads the plate text
- **FastAPI** exposes a clean HTTP API for web/mobile clients

> Input: raw image (car, parking lot, gate camera)  
> Output: list of plates with bounding boxes + detection & OCR confidences

---

## ✨ What this project does

- ✅ Detects license plates in images using a **fine-tuned YOLOv8n** detector  
- ✅ Crops detected plates using **PIL (Pillow)**  
- ✅ Runs **Tesseract OCR** with a custom `lpr` model trained on cropped plates  
- ✅ Serves everything via **FastAPI** (`/v1/plates/detect`) as a JSON API  
- ✅ Designed so it can plug into a **web UI / mobile app / gate controller**

This is not just a toy notebook – it’s a **minimal but realistic LPR backend**.

---
🧱 Tech Stack

Detection: YOLOv8n (Ultralytics) – transfer-learning from COCO, fine-tuned on a license plate dataset (Pascal VOC → YOLO format).
OCR: Tesseract OCR (eng → fine-tuned lpr using tesstrain).
Backend API: FastAPI + Uvicorn.
Data processing: Python, Pillow, PyTorch, pytesseract.
Environment: Python 3.12, virtualenv, Ubuntu (but should run anywhere Tesseract + Python run).

---

## 🧠 High-Level Architecture

**Data flow for one request:**

1. Client uploads an image (e.g., JPEG from a camera).
2. FastAPI endpoint `/v1/plates/detect`:
   - Loads image with Pillow.
   - Runs YOLOv8n to detect license plate bounding boxes.
   - For each box, crops the plate region.
   - Runs Tesseract with `--psm 7 -l lpr` on each crop.
3. Response JSON:

```json
{
  "image_width": 1280,
  "image_height": 720,
  "num_detections": 2,
  "detections": [
    {
      "bbox": {"x1": 432, "y1": 310, "x2": 615, "y2": 360},
      "detection_confidence": 0.93,
      "plate_text": "ABC123",
      "ocr_confidence": 88.5
    },
    {
      "bbox": {"x1": 850, "y1": 305, "x2": 1020, "y2": 355},
      "detection_confidence": 0.90,
      "plate_text": "BHL9021",
      "ocr_confidence": 84.2
    }
  ]
}

🧱 Tech Stack
