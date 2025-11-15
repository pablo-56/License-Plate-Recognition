import pandas as pd
from pathlib import Path
from PIL import Image
import shutil

HOME = Path.home()
root = HOME / "Desktop" / "lpd"

plates_dir = root / "data" / "ocr" / "plates"
csv_path = plates_dir / "plates.csv"

df = pd.read_csv(csv_path)

# Path to tesstrain repo (you will clone it later)
tesstrain_root = HOME / "Desktop" / "tesstrain"
gt_dir = tesstrain_root / "data" / "lpr-ground-truth"
gt_dir.mkdir(parents=True, exist_ok=True)

for _, row in df.iterrows():
    fname = row["filename"]
    text = str(row["plate_text"]).strip()

    if not text:
        continue  # skip unlabeled ones for now

    src_img = plates_dir / fname
    if not src_img.exists():
        continue

    # Copy/convert image to tesstrain GT dir
    # (PNG is fine; TIFF also okay)
    img = Image.open(src_img).convert("L")  # grayscale is good enough
    dst_img = gt_dir / fname
    img.save(dst_img)

    # Create .gt.txt file with plate text
    gt_txt = gt_dir / (dst_img.stem + ".gt.txt")
    gt_txt.write_text(text + "\n")

print("Ground-truth files in:", gt_dir)
