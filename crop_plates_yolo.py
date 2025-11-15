from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import csv

HOME = Path.home()
root = HOME / "Desktop" / "lpd"

# YOLO weights (your trained detector)
weights = root / "notebooks" / "runs" / "detect" / "lpr-car-plate-yolov8n-cpu" / "weights" / "best.pt"
model = YOLO(str(weights))

# Where your original images live (train split)
images_dir = root / "data" / "yolo" / "images" / "train"

# Output directory for plate crops
plates_dir = root / "data" / "ocr" / "plates"
plates_dir.mkdir(parents=True, exist_ok=True)

csv_path = plates_dir / "plates.csv"

rows = []
crop_id = 0

for img_path in images_dir.glob("*.*"):
    im = Image.open(img_path).convert("RGB")
    w, h = im.size

    results = model.predict(
        source=str(img_path),
        device="cpu",   # GPU if you want, but CPU is fine here
        imgsz=320,
        conf=0.4,      # our F1-optimal threshold
        verbose=False,
    )[0]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        # clamp to image bounds
        x1, y1, x2, y2 = map(int, [max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)])

        crop = im.crop((x1, y1, x2, y2))

        crop_name = f"plate_{crop_id:05d}.png"
        crop.save(plates_dir / crop_name)
        crop_id += 1

        rows.append({"filename": crop_name, "plate_text": ""})  # fill later

# Write CSV mapping crop -> (empty) text
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "plate_text"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} crops to {plates_dir}")
print(f"CSV skeleton: {csv_path}")
