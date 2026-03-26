"""
VisionExtract Configuration
All hyperparameters and paths in one place.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "coco")
OUTPUTS_DIR   = os.path.join(BASE_DIR, "outputs")
CKPT_DIR      = os.path.join(OUTPUTS_DIR, "checkpoints")
PRED_DIR      = os.path.join(OUTPUTS_DIR, "predictions")
LOG_DIR       = os.path.join(OUTPUTS_DIR, "logs")

# COCO split paths  (edit these to point at your downloaded COCO folder)
# Using val2017 for both train and val (only val images downloaded)
TRAIN_IMG_DIR   = os.path.join(DATA_DIR, "images", "val2017")
VAL_IMG_DIR     = os.path.join(DATA_DIR, "images", "val2017")
TRAIN_ANN_FILE  = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")
VAL_ANN_FILE    = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")

# ─── Image / Mask settings ────────────────────────────────────────────────────
IMAGE_SIZE  = 256           # reduced for faster training on MPS/CPU
NUM_CLASSES = 1             # binary: subject vs background

# ─── Model ────────────────────────────────────────────────────────────────────
ENCODER       = "resnet50"  # timm / torchvision backbone
ENCODER_WEIGHTS = "imagenet"
ARCHITECTURE  = "Unet"      # or "DeepLabV3Plus", "FPN", "PAN"

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE     = 4
NUM_WORKERS    = 2
EPOCHS         = 30
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
SCHEDULER      = "cosine"   # "cosine" | "plateau"

# Loss weights
BCE_WEIGHT  = 0.5
DICE_WEIGHT = 0.5

# ─── Augmentation ─────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]   # ImageNet stats
STD  = [0.229, 0.224, 0.225]

# ─── Evaluation ───────────────────────────────────────────────────────────────
IOU_THRESHOLD = 0.5

# ─── Inference ────────────────────────────────────────────────────────────────
BEST_CKPT = os.path.join(CKPT_DIR, "best_model.pth")
