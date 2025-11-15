from ultralytics import YOLO
from pathlib import Path

HOME = Path.home()
root = HOME / "Desktop" / "lpd"
data_yaml = root / "data" / "yolo" / "lpr_car_plate.yaml"

print("Using dataset config:", data_yaml)

model = YOLO("yolov8n.pt")

model.train(
    data=str(data_yaml),
    epochs=50,
    imgsz=416,
    batch=8,        # you can afford a bit larger batch on CPU RAM
    device="cpu",   # <--- force CPU
    workers=0,
    cache=False,
    name="lpr-car-plate-yolov8n-cpu",
)
