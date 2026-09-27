import os
import argparse
import random
import cv2
from ultralytics import YOLO

import config


def run_inference(weights_path, n_images=12):
    test_dir = os.path.join(config.DATA_DIR, config.TEST_IMAGES_SUBDIR)
    out_dir = os.path.join(config.BASE_DIR, "test_predictions")
    os.makedirs(out_dir, exist_ok=True)

    model = YOLO(weights_path)

    all_images = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    sample = random.sample(all_images, min(n_images, len(all_images)))

    for img_name in sample:
        img_path = os.path.join(test_dir, img_name)
        results = model.predict(
            img_path,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            verbose=False,
        )
        annotated = results[0].plot()  # returns numpy BGR image with boxes+masks+labels drawn
        out_path = os.path.join(out_dir, f"pred_{img_name}")
        cv2.imwrite(out_path, annotated)
        print(f"Saved: {out_path}  ({len(results[0].boxes)} detections)")

    print(f"\nAll annotated test predictions saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=os.path.join(config.RUNS_DIR, config.RUN_NAME, "weights", "best.pt"),
    )
    parser.add_argument("--n", type=int, default=12, help="number of test images to sample")
    args = parser.parse_args()

    run_inference(args.weights, n_images=args.n)


if __name__ == "__main__":
    main()
