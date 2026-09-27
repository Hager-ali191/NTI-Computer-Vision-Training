"""
07_export_model.py

Exports the fine-tuned .pt model to portable inference formats
(ONNX by default; TorchScript / OpenVINO / TensorRT also supported by
Ultralytics with a one-line format change).

Usage:
    python 07_export_model.py --weights runs/instance_seg_finetune/weights/best.pt --format onnx
"""

import os
import argparse
import shutil
from ultralytics import YOLO

import config


def export(weights_path, fmt="onnx"):
    os.makedirs(config.EXPORT_DIR, exist_ok=True)
    model = YOLO(weights_path)
    exported_path = model.export(format=fmt, imgsz=config.IMG_SIZE)
    dest = os.path.join(config.EXPORT_DIR, os.path.basename(exported_path))
    shutil.copy(exported_path, dest)
    print(f"Exported model ({fmt}) -> {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=os.path.join(config.RUNS_DIR, config.RUN_NAME, "weights", "best.pt"),
    )
    parser.add_argument("--format", default="onnx",
                        choices=["onnx", "torchscript", "openvino", "engine", "coreml"])
    args = parser.parse_args()
    export(args.weights, fmt=args.format)


if __name__ == "__main__":
    main()
