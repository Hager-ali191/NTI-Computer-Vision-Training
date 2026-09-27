# COCO Instance Segmentation Pipeline (Roboflow + YOLOv8-seg)

End-to-end pipeline: download → analyze → preprocess/validate → fine-tune → evaluate → test → export.

Model choice: **YOLOv8-seg** (Ultralytics), fine-tuned from COCO-pretrained weights.
It's the natural fit here because Roboflow exports directly to YOLOv8-seg format,
training/eval/export are all one library, and it performs strongly on instance
segmentation without the setup overhead of Detectron2/Mask R-CNN.

## 0. Setup

```bash
pip install -r requirements.txt
```

Set your Roboflow API key (find it in Roboflow: Settings → API Keys):

```bash
export ROBOFLOW_API_KEY="your_key_here"
```

Edit `config.py`:
- `ROBOFLOW_WORKSPACE`, `ROBOFLOW_PROJECT`, `ROBOFLOW_VERSION` → your dataset
- `BASE_MODEL` → yolov8n/s/m/l/x-seg.pt (bigger = more accurate, slower)
- `EPOCHS`, `BATCH_SIZE`, `IMG_SIZE`, `DEVICE` → your hardware/time budget

## 1. Download the dataset

```bash
python 01_download_dataset.py
```

Pulls **two** copies of your Roboflow dataset:
- `dataset/` — YOLOv8-seg format (polygon labels) → used for training
- `dataset_coco/` — COCO JSON format → used for EDA and independent COCOeval

## 2. Analyze the dataset (EDA)

```bash
python 02_dataset_analysis.py
```

Produces (in `eda_outputs/`):
- split sizes (train/valid/test image & annotation counts)
- class distribution per split (bar chart + CSV)
- object size distribution (COCO small/medium/large buckets)
- instances-per-image histogram
- sample images with masks + boxes overlaid

Use this to catch class imbalance, near-empty classes, or mislabeled data
**before** you spend GPU time training.

## 3. Preprocess & validate

```bash
python 03_preprocess_validate.py         # report only
python 03_preprocess_validate.py --fix   # also remove unreadable images
```

Checks:
- every image has a matching label file and vice versa
- label files aren't empty
- polygon coordinates are well-formed (even count, normalized to [0,1])
- flags class imbalance (max/min instance ratio > 5x)

## 4. Fine-tune

```bash
python 04_train.py
```

- loads COCO-pretrained `BASE_MODEL` weights
- freezes the first `FREEZE_LAYERS` backbone layers (fast convergence, less
  overfitting on small datasets) — set `FREEZE_LAYERS = 0` in config.py to
  fine-tune everything
- applies augmentation (mosaic, mixup, HSV jitter, affine transforms)
- early stops after `PATIENCE` epochs without improvement
- saves best/last weights + training curves + confusion matrix under
  `runs/instance_seg_finetune/`

## 5. Evaluate

```bash
python 05_evaluate.py --weights runs/instance_seg_finetune/weights/best.pt
```

Two independent evaluations:
1. **Ultralytics val()** on the `valid` split → box mAP50/mAP50-95, mask
   mAP50/mAP50-95, per-class precision/recall, PR curves, confusion matrix.
2. **pycocotools COCOeval** on the `test` split (fully independent of the
   training framework's metric code) → standard COCO AP/AR table for both
   bbox and segmentation.

## 6. Test / visualize predictions

```bash
python 06_test_inference.py --weights runs/instance_seg_finetune/weights/best.pt --n 12
```

Runs inference on a random sample of test images and saves annotated
images (boxes + masks + labels + confidence) to `test_predictions/`.

## 7. Export for deployment

```bash
python 07_export_model.py --weights runs/instance_seg_finetune/weights/best.pt --format onnx
```

Supported formats: `onnx`, `torchscript`, `openvino`, `engine` (TensorRT), `coreml`.

## Notes / tuning tips

- **Small dataset (<500 images):** start from `yolov8n-seg.pt` or
  `yolov8s-seg.pt`, keep `FREEZE_LAYERS` high (10+), rely heavily on
  augmentation, expect more epochs to help since the model won't overfit
  as fast when frozen.
- **Class imbalance:** if `03_preprocess_validate.py` flags it, either
  collect more data for rare classes or upsample those images in
  `data.yaml`'s train list.
- **mAP50-95 much lower than mAP50:** usually means bounding boxes/masks
  are roughly right but not tight — check annotation quality and increase
  `IMG_SIZE` if objects are small.
- **GPU memory errors:** lower `BATCH_SIZE` or `IMG_SIZE` in `config.py`.
