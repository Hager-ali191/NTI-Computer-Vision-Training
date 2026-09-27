import os

from roboflow import Roboflow


ROBOFLOW_API_KEY = os.environ.get("bbrsCDbFqgCpWkPYiqdt", "rf_qEveYiE2oLgHt46wAKh3m5nKRIi2")
ROBOFLOW_WORKSPACE = "microsoft"    
ROBOFLOW_PROJECT = "coco-dataset-vdnr1"     
ROBOFLOW_VERSION = 41                   

# ROBOFLOW_FORMAT_YOLO = "yolov8"
ROBOFLOW_FORMAT_COCO = "coco-segmentation"


# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")           # YOLO-format dataset (for training)
COCO_DATA_DIR = os.path.join(BASE_DIR, "dataset_coco")  # COCO-format dataset (for EDA/eval)
RUNS_DIR = os.path.join(BASE_DIR, "runs")
EXPORT_DIR = os.path.join(BASE_DIR, "exported_model")

# MODEL TRAINING
# Pretrained checkpoint to fine-tune from. Options (small -> large):
# yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt
BASE_MODEL = "yolov8s-seg.pt"

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 20            # early stopping patience (epochs with no improvement)
DEVICE = 0               # 0 for first GPU, "cpu" for CPU, "0,1" for multi-GPU
WORKERS = 8
OPTIMIZER = "auto"        # "SGD", "Adam", "AdamW", or "auto"
LR0 = 0.01                 # initial learning rate
SEED = 42

# Fine-tuning strategy: freeze the first N backbone layers to preserve
# pretrained features and speed up convergence on small datasets. Set to 0
# to fine-tune the whole network.
FREEZE_LAYERS = 10

RUN_NAME = "instance_seg_finetune"

# ------------------------------------------------------------------
# EVALUATION / INFERENCE SETTINGS
# ------------------------------------------------------------------
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
TEST_IMAGES_SUBDIR = "test/images"
