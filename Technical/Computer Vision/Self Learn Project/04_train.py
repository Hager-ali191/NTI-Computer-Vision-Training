"""
04_train.py

Fine-tunes a pretrained YOLOv8 segmentation model on the Roboflow dataset.

Usage:
    python 04_train.py
"""

import os
import yaml
from ultralytics import YOLO

import config


def patch_data_yaml_paths():
    """Roboflow's data.yaml sometimes uses relative paths that break outside
    the original download context. Rewrite train/val/test paths as absolute."""
    yaml_path = os.path.join(config.DATA_DIR, "data.yaml")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    for split_key, subdir in [("train", "train/images"), ("val", "valid/images"), ("test", "test/images")]:
        candidate = os.path.join(config.DATA_DIR, subdir)
        if os.path.isdir(candidate):
            data[split_key] = candidate

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f)

    print(f"Patched data.yaml paths:\n{data}")
    return yaml_path


def train():
    data_yaml = patch_data_yaml_paths()

    model = YOLO(config.BASE_MODEL)  # loads pretrained COCO weights

    results = model.train(
        data=data_yaml,
        epochs=config.EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        device=config.DEVICE,
        workers=config.WORKERS,
        optimizer=config.OPTIMIZER,
        lr0=config.LR0,
        patience=config.PATIENCE,
        seed=config.SEED,
        freeze=config.FREEZE_LAYERS,
        project=config.RUNS_DIR,
        name=config.RUN_NAME,
        # augmentation (tune as needed for your domain)
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        val=True,
        plots=True,
        save=True,
        exist_ok=True,
    )

    best_weights = os.path.join(config.RUNS_DIR, config.RUN_NAME, "weights", "best.pt")
    print(f"\nTraining complete. Best weights saved at:\n  {best_weights}")
    return best_weights


if __name__ == "__main__":
    train()
