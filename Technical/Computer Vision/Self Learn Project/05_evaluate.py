"""
05_evaluate.py

Two-stage evaluation of the fine-tuned model:
  1. Ultralytics' built-in val() -> box + mask mAP50, mAP50-95, precision/recall
     per class, confusion matrix, PR curves (saved as plots automatically).
  2. Independent COCO-style evaluation using pycocotools, run against the
     COCO-format test split, for an apples-to-apples mAP number and a
     per-class AP breakdown.

Usage:
    python 05_evaluate.py --weights runs/instance_seg_finetune/weights/best.pt
"""

import os
import json
import argparse
import cv2
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import config


def ultralytics_val(weights_path, split="val"):
    print(f"\n=== Ultralytics built-in validation ({split}) ===")
    model = YOLO(weights_path)
    metrics = model.val(
        data=os.path.join(config.DATA_DIR, "data.yaml"),
        split=split,
        imgsz=config.IMG_SIZE,
        conf=config.CONF_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        plots=True,
        save_json=True,
    )
    print("Box mAP50-95:", metrics.box.map)
    print("Box mAP50   :", metrics.box.map50)
    print("Mask mAP50-95:", metrics.seg.map)
    print("Mask mAP50   :", metrics.seg.map50)
    return metrics


def predictions_to_coco_json(weights_path, coco_gt, img_dir, out_json):
    """Run inference over every image in the COCO test split and dump
    predictions in COCO results format for COCOeval."""
    model = YOLO(weights_path)
    results_coco = []

    img_ids = coco_gt.getImgIds()
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        preds = model.predict(
            img_path, conf=config.CONF_THRESHOLD, iou=config.IOU_THRESHOLD, verbose=False
        )[0]

        if preds.masks is None:
            continue

        cat_id_map = {i: c["id"] for i, c in enumerate(coco_gt.loadCats(coco_gt.getCatIds()))}

        for box, score, cls, mask_poly in zip(
            preds.boxes.xywh.cpu().numpy(),
            preds.boxes.conf.cpu().numpy(),
            preds.boxes.cls.cpu().numpy().astype(int),
            preds.masks.xy,
        ):
            x_c, y_c, w, h = box
            x = x_c - w / 2
            y = y_c - h / 2
            segmentation = mask_poly.flatten().tolist()
            results_coco.append({
                "image_id": img_id,
                "category_id": cat_id_map.get(cls, cls),
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(score),
                "segmentation": [segmentation],
            })

    with open(out_json, "w") as f:
        json.dump(results_coco, f)
    print(f"Saved {len(results_coco)} predictions -> {out_json}")
    return out_json


def coco_style_eval(weights_path, split="test"):
    print(f"\n=== Independent COCOeval ({split}) ===")
    ann_path = os.path.join(config.COCO_DATA_DIR, split, "_annotations.coco.json")
    img_dir = os.path.join(config.COCO_DATA_DIR, split)
    if not os.path.exists(ann_path):
        print(f"[warn] no COCO annotations at {ann_path}, skipping")
        return

    coco_gt = COCO(ann_path)
    pred_json = os.path.join(config.BASE_DIR, f"predictions_{split}.json")
    predictions_to_coco_json(weights_path, coco_gt, img_dir, pred_json)

    coco_dt = coco_gt.loadRes(pred_json)

    for iou_type in ["bbox", "segm"]:
        print(f"\n--- {iou_type.upper()} metrics ---")
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=os.path.join(config.RUNS_DIR, config.RUN_NAME, "weights", "best.pt"),
    )
    args = parser.parse_args()

    ultralytics_val(args.weights, split="val")
    coco_style_eval(args.weights, split="test")


if __name__ == "__main__":
    main()
