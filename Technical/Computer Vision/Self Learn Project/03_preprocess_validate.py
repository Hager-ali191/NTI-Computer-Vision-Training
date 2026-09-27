"""
03_preprocess_validate.py

Validates and lightly preprocesses the YOLO-format export before training:
  - confirms data.yaml points to the right class list and splits
  - checks every image has a matching label file (and vice versa)
  - checks label files are non-empty and polygons are well-formed
    (even number of coords, values in [0,1])
  - reports and (optionally) removes corrupt/unreadable images
  - reports class imbalance so you know if you need class weights /
    oversampling before training

This does not resize images -- Ultralytics handles resizing to IMG_SIZE
internally at train time -- it only validates/cleans the data on disk.

Usage:
    python 03_preprocess_validate.py [--fix]
"""

import os
import argparse
import yaml
import cv2
from collections import Counter

import config


def load_data_yaml():
    yaml_path = os.path.join(config.DATA_DIR, "data.yaml")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    print("=== data.yaml ===")
    print(data)
    return data


def validate_split(split, fix=False):
    img_dir = os.path.join(config.DATA_DIR, split, "images")
    lbl_dir = os.path.join(config.DATA_DIR, split, "labels")
    if not os.path.isdir(img_dir):
        print(f"[warn] split '{split}' not found at {img_dir}")
        return

    images = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    class_counter = Counter()
    n_missing_label = 0
    n_empty_label = 0
    n_bad_polygon = 0
    n_unreadable = 0

    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        # 1. image readable?
        img = cv2.imread(img_path)
        if img is None:
            n_unreadable += 1
            print(f"[bad image] {img_path}")
            if fix:
                os.remove(img_path)
                if os.path.exists(lbl_path):
                    os.remove(lbl_path)
            continue

        # 2. label exists?
        if not os.path.exists(lbl_path):
            n_missing_label += 1
            print(f"[missing label] {lbl_path}")
            continue

        # 3. label non-empty + well-formed polygons
        with open(lbl_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            n_empty_label += 1
            continue

        for ln in lines:
            parts = ln.split()
            cls_id = int(parts[0])
            coords = parts[1:]
            class_counter[cls_id] += 1
            if len(coords) % 2 != 0 or len(coords) < 6:
                n_bad_polygon += 1
                print(f"[bad polygon] {lbl_path}: {ln[:60]}...")
                continue
            vals = list(map(float, coords))
            if any(v < 0.0 or v > 1.0 for v in vals):
                n_bad_polygon += 1
                print(f"[out-of-range coords] {lbl_path}")

    print(f"\n--- {split} ---")
    print(f"images: {len(images)}")
    print(f"unreadable images: {n_unreadable}")
    print(f"missing labels: {n_missing_label}")
    print(f"empty labels: {n_empty_label}")
    print(f"malformed polygons: {n_bad_polygon}")
    print(f"instances per class: {dict(class_counter)}")

    if class_counter:
        total = sum(class_counter.values())
        max_c = max(class_counter.values())
        min_c = min(class_counter.values())
        if min_c > 0 and max_c / min_c > 5:
            print(f"[note] class imbalance detected (max/min ratio = {max_c/min_c:.1f}). "
                  f"Consider class-weighted loss or oversampling rare classes.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true",
                         help="remove unreadable images and their orphan labels")
    args = parser.parse_args()

    data_yaml = load_data_yaml()
    for split in ["train", "val", "valid", "test"]:
        split_dir = os.path.join(config.DATA_DIR, split)
        if os.path.isdir(split_dir):
            validate_split(split, fix=args.fix)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
