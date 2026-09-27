"""
02_dataset_analysis.py

Exploratory data analysis on the COCO-format export:
  - split sizes (train/valid/test)
  - class distribution (instance counts per class)
  - images per class
  - object size distribution (small/medium/large, per COCO definition)
  - instances per image distribution
  - sample annotated images (masks + boxes overlaid)

Outputs figures to ./eda_outputs/

Usage:
    python 02_dataset_analysis.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

import config

OUT_DIR = os.path.join(config.BASE_DIR, "eda_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SPLITS = ["train", "valid", "test"]


def load_coco_splits():
    cocos = {}
    for split in SPLITS:
        ann_path = os.path.join(config.COCO_DATA_DIR, split, "_annotations.coco.json")
        if os.path.exists(ann_path):
            cocos[split] = COCO(ann_path)
        else:
            print(f"[warn] missing annotations for split '{split}' at {ann_path}")
    return cocos


def coco_area_bucket(area):
    # Official COCO thresholds (in pixel^2)
    if area < 32 ** 2:
        return "small"
    elif area < 96 ** 2:
        return "medium"
    return "large"


def split_sizes(cocos):
    rows = []
    for split, coco in cocos.items():
        rows.append({
            "split": split,
            "images": len(coco.getImgIds()),
            "annotations": len(coco.getAnnIds()),
            "categories": len(coco.getCatIds()),
        })
    df = pd.DataFrame(rows)
    print("\n=== Split sizes ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT_DIR, "split_sizes.csv"), index=False)
    return df


def class_distribution(cocos):
    records = []
    for split, coco in cocos.items():
        cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
        for ann in coco.loadAnns(coco.getAnnIds()):
            records.append({
                "split": split,
                "class": cats[ann["category_id"]],
                "area": ann["area"],
                "image_id": ann["image_id"],
            })
    df = pd.DataFrame(records)
    if df.empty:
        print("[warn] no annotations found across splits")
        return df

    # Instance counts per class per split
    counts = df.groupby(["split", "class"]).size().reset_index(name="instance_count")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=counts, x="class", y="instance_count", hue="split")
    plt.xticks(rotation=45, ha="right")
    plt.title("Instance count per class per split")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    # Object size buckets
    df["size_bucket"] = df["area"].apply(coco_area_bucket)
    size_counts = df.groupby(["split", "size_bucket"]).size().reset_index(name="count")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=size_counts, x="size_bucket", y="count", hue="split",
                order=["small", "medium", "large"])
    plt.title("Object size distribution (COCO small/medium/large)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "object_size_distribution.png"), dpi=150)
    plt.close()

    print("\n=== Instance counts per class ===")
    print(counts.pivot(index="class", columns="split", values="instance_count").fillna(0))

    counts.to_csv(os.path.join(OUT_DIR, "class_counts.csv"), index=False)
    return df


def instances_per_image(cocos, split="train"):
    if split not in cocos:
        return
    coco = cocos[split]
    img_ids = coco.getImgIds()
    counts = [len(coco.getAnnIds(imgIds=i)) for i in img_ids]
    plt.figure(figsize=(8, 5))
    sns.histplot(counts, bins=30)
    plt.title(f"Instances per image ({split} split)")
    plt.xlabel("number of instances")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"instances_per_image_{split}.png"), dpi=150)
    plt.close()
    print(f"\n{split}: mean instances/image = {np.mean(counts):.2f}, "
          f"max = {np.max(counts)}, images with 0 instances = {sum(c == 0 for c in counts)}")


def visualize_samples(cocos, split="train", n=6):
    if split not in cocos:
        return
    coco = cocos[split]
    img_dir = os.path.join(config.COCO_DATA_DIR, split)
    img_ids = coco.getImgIds()
    sample_ids = random.sample(img_ids, min(n, len(img_ids)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, img_id in zip(axes, sample_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            ax.set_title("missing image")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        overlay = img.copy()
        for ann in anns:
            if "segmentation" in ann and ann["segmentation"]:
                if isinstance(ann["segmentation"], list):
                    rles = maskUtils.frPyObjects(ann["segmentation"], img_info["height"], img_info["width"])
                    rle = maskUtils.merge(rles)
                else:
                    rle = ann["segmentation"]
                m = maskUtils.decode(rle)
                color = np.random.randint(0, 255, 3)
                overlay[m == 1] = overlay[m == 1] * 0.5 + color * 0.5
            x, y, w, h = ann["bbox"]
            cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(f"img {img_id} ({len(anns)} instances)")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"sample_annotations_{split}.png"), dpi=150)
    plt.close()
    print(f"Saved sample annotation visualization -> {OUT_DIR}/sample_annotations_{split}.png")


def main():
    cocos = load_coco_splits()
    if not cocos:
        print("No COCO-format splits found. Run 01_download_dataset.py first.")
        return

    split_sizes(cocos)
    class_distribution(cocos)
    for split in cocos:
        instances_per_image(cocos, split)
    visualize_samples(cocos, split="train")

    print(f"\nAll EDA artifacts saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
