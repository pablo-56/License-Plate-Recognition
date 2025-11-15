from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
import random


# -----------------------------
#  Paths (adapted to your setup)
# -----------------------------
HOME = Path.home()
root = HOME / "Desktop" / "lpd"

cp_dir = root / "data" / "raw" / "car_plate"
annotations_dir = cp_dir / "annotations"
images_dir = cp_dir / "images"

yolo_base = root / "data" / "yolo"
img_out_dirs = {
    "train": yolo_base / "images" / "train",
    "val":   yolo_base / "images" / "val",
    "test":  yolo_base / "images" / "test",
}
lbl_out_dirs = {
    "train": yolo_base / "labels" / "train",
    "val":   yolo_base / "labels" / "val",
    "test":  yolo_base / "labels" / "test",
}

# Create output directories
for d in list(img_out_dirs.values()) + list(lbl_out_dirs.values()):
    d.mkdir(parents=True, exist_ok=True)


# -----------------------------
#  Helpers
# -----------------------------

def voc_box_to_yolo_line(xmin, ymin, xmax, ymax, img_w, img_h, class_id=0):
    """
    Convert Pascal VOC box (absolute pixels) → YOLO format (normalized).
    """
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def parse_voc_xml(xml_path: Path):
    """
    Parse a single Pascal VOC XML file.
    Returns:
      - image filename (string)
      - image width, height (ints)
      - list of dicts: [{"name": class_name, "xmin":..., "ymin":..., "xmax":..., "ymax":...}, ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    img_w = int(size.findtext("width"))
    img_h = int(size.findtext("height"))

    objects = []
    for obj in root.findall("object"):
        cls_name = obj.findtext("name")
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))
        objects.append(
            {
                "name": cls_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return filename, img_w, img_h, objects


# -----------------------------
#  Split config
# -----------------------------
# Adjust these if you want a different split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-6, "Ratios must sum to 1.0"


# -----------------------------
#  Main conversion
# -----------------------------
def main():
    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise RuntimeError(f"No XML files found in {annotations_dir}")

    random.seed(42)
    random.shuffle(xml_files)

    n_total = len(xml_files)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    # rest goes to test

    print(f"Total annotations: {n_total}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_total - n_train - n_val}")

    for idx, xml_path in enumerate(xml_files):
        # Decide split
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "test"

        img_out_dir = img_out_dirs[split]
        lbl_out_dir = lbl_out_dirs[split]

        filename, img_w, img_h, objects = parse_voc_xml(xml_path)

        # Find corresponding image
        img_path = images_dir / filename
        if not img_path.exists():
            # Some datasets store different extensions (.jpg vs .png), try a few
            alt = None
            for ext in [".jpg", ".jpeg", ".png"]:
                cand = images_dir / (Path(filename).stem + ext)
                if cand.exists():
                    alt = cand
                    break
            if alt is None:
                print(f"[WARN] Image not found for XML {xml_path.name}, skipping.")
                continue
            img_path = alt

        # Copy image to target split
        dst_img_path = img_out_dir / img_path.name
        shutil.copy2(img_path, dst_img_path)

        # Build YOLO label lines
        yolo_lines = []
        for obj in objects:
            # Map ALL objects to class 0 "license-plate" (you can add a mapping here if needed)
            class_id = 0
            xmin = obj["xmin"]
            ymin = obj["ymin"]
            xmax = obj["xmax"]
            ymax = obj["ymax"]
            line = voc_box_to_yolo_line(xmin, ymin, xmax, ymax, img_w, img_h, class_id)
            yolo_lines.append(line)

        # Write label file (same stem as image)
        dst_lbl_path = lbl_out_dir / (dst_img_path.stem + ".txt")
        dst_lbl_path.write_text("\n".join(yolo_lines))

    print("Conversion complete.")
    print(f"YOLO dataset created under: {yolo_base}")


if __name__ == "__main__":
    main()
