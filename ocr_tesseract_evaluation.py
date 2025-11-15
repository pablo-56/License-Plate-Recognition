import pandas as pd
from pathlib import Path
import subprocess

HOME = Path.home()
root = HOME / "Desktop" / "lpd"

plates_dir = root / "data" / "ocr" / "plates"
csv_path = plates_dir / "plates.csv"

df = pd.read_csv(csv_path)
df = df[df["plate_text"].notna() & (df["plate_text"].str.strip() != "")]

def run_tesseract(img_path: Path, lang: str = "lpr") -> str:
    result = subprocess.run(
        ["tesseract", str(img_path), "stdout", "--psm", "7", "-l", lang],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()

y_true = []
y_pred = []

for _, row in df.iterrows():
    fname = row["filename"]
    gt = row["plate_text"].strip()
    img_path = plates_dir / fname

    pred = run_tesseract(img_path, lang="lpr")  # use eng to compare baseline
    y_true.append(gt)
    y_pred.append(pred)

# Compute simple plate-level accuracy
correct = sum(p == t for p, t in zip(y_pred, y_true))
acc = correct / len(y_true)

print(f"Plate-level accuracy: {acc:.3f} ({correct}/{len(y_true)})")

# (Optional) character-level accuracy
total_chars = sum(len(t) for t in y_true)
char_correct = sum(
    sum(1 for pc, tc in zip(p, t) if pc == tc)
    for p, t in zip(y_pred, y_true)
)
print(f"Char-level accuracy (rough): {char_correct / total_chars:.3f}")
