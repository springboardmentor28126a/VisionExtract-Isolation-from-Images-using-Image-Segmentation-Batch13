# VisionExtract — AI Subject Isolation from Images

> Automatically extract the main subject from any image. Background pixels are rendered black.
> Built with PyTorch · U-Net + ResNet-50 · COCO 2017

---

## Project Overview

| Item | Detail |
|------|--------|
| Task | Binary semantic segmentation (subject vs. background) |
| Dataset | [COCO 2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) |
| Model | U-Net with ResNet-50 ImageNet-pretrained encoder |
| Loss | BCE (0.5) + Dice (0.5) combined loss |
| Metrics | IoU, Dice Coefficient, Pixel Accuracy, Precision, Recall |
| Output | Isolated-subject image (background = black) |

---

## Project Structure

```
visionextract/
├── config.py                  ← All hyperparameters & paths
├── requirements.txt           ← Python dependencies
├── train.py                   ← Training loop (Milestone 2–3)
├── evaluate.py                ← Full evaluation report (Milestone 3)
├── inference.py               ← CLI inference on images/folders (Milestone 3)
├── app.py                     ← Gradio web application (Milestone 4)
│
├── data/
│   ├── coco_dataset.py        ← COCO loader + binary mask builder
│   └── transforms.py          ← Train / Val / Inference augmentations
│
├── models/
│   └── segmentation_model.py  ← U-Net / DeepLabV3+ factory + checkpoint loader
│
├── utils/
│   ├── losses.py              ← DiceLoss + CombinedLoss
│   ├── metrics.py             ← IoU, Dice, PixelAcc, Precision, Recall
│   └── visualize.py           ← Mask overlay, comparison grids
│
└── outputs/
    ├── checkpoints/           ← best_model.pth, last_model.pth
    ├── predictions/           ← Sample grids + inference results
    └── logs/                  ← TensorBoard event files
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone <repo>
cd visionextract
pip install -r requirements.txt
```

### 2. Download COCO 2017

Download from Kaggle: https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset

Extract so the directory looks like:
```
data/coco/
├── images/
│   ├── train2017/   ← ~118k images
│   └── val2017/     ← ~5k images
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

Then update `config.py` → `DATA_DIR` if needed (default: `data/coco` relative to project root).

---

## Milestone Workflow

### Milestone 1 — Dataset Acquisition (Week 1–2)

```bash
# Inspect COCO structure
python -c "
from data.coco_dataset import build_datasets
train_ds, val_ds = build_datasets(max_train=100, max_val=20)
img, mask = train_ds[0]
print('Image:', img.shape, 'Mask:', mask.shape)
"
```

### Milestone 2 — Initial Training (Week 3–4)

```bash
# Quick test run (5k images, 5 epochs)
python train.py --max_train 5000 --max_val 500 --epochs 5

# Full training
python train.py --epochs 30

# Resume interrupted training
python train.py --resume outputs/checkpoints/last_model.pth
```

**TensorBoard monitoring:**
```bash
tensorboard --logdir outputs/logs
```

### Milestone 3 — Evaluation & Fine-tuning (Week 5–6)

```bash
# Full evaluation report
python evaluate.py --save_grids

# Quick eval on 1000 images
python evaluate.py --max_val 1000

# Experiment with DeepLabV3+
python train.py --arch DeepLabV3Plus --epochs 30
```

### Milestone 4 — Inference & Web UI (Week 7–8)

```bash
# Single image
python inference.py --input photo.jpg --show --overlay

# Batch folder
python inference.py --input /path/to/images --output /path/to/results

# Launch web app
python app.py

# Public share link (for demo)
python app.py --share
```

---

## Configuration (`config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_SIZE` | 512 | Input image size (H=W) |
| `ARCHITECTURE` | `"Unet"` | `Unet`, `UnetPlusPlus`, `DeepLabV3Plus`, `FPN`, `PAN` |
| `ENCODER` | `"resnet50"` | Any timm/torchvision backbone |
| `BATCH_SIZE` | 8 | Per-GPU batch size |
| `EPOCHS` | 30 | Total training epochs |
| `LR` | 1e-4 | Adam learning rate |
| `BCE_WEIGHT` | 0.5 | Weight of BCE in combined loss |
| `DICE_WEIGHT` | 0.5 | Weight of Dice in combined loss |
| `SCHEDULER` | `"cosine"` | `cosine` or `plateau` |

---

## How Binary Masks are Built

COCO contains 80 object categories with instance-level polygon annotations.
This project converts them to a **single binary mask per image**:

```
binary_mask = union of all instance masks
            = 1 where any annotated object exists
            = 0 for pure background
```

This trains the model to isolate *any foreground object* regardless of category.

---

## Model Architecture

```
Input (3 × 512 × 512)
        │
   ResNet-50 Encoder (ImageNet pretrained)
   [64 → 256 → 512 → 1024 → 2048 features]
        │
   U-Net Skip Connections (encoder features fused with decoder)
        │
   Decoder (upsampling + conv blocks)
        │
   Output head (1 × 512 × 512 logits)
        │
   Sigmoid → probability map
        │
   Threshold (0.5) → binary mask
```

---

## Expected Results (COCO 2017 Val)

| Metric | Expected Range |
|--------|---------------|
| IoU | 0.70 – 0.82 |
| Dice | 0.78 – 0.88 |
| Pixel Accuracy | 0.90 – 0.96 |
| Precision | 0.80 – 0.90 |
| Recall | 0.82 – 0.92 |

> Results vary based on architecture, training duration, and augmentation settings.

---

## Evaluation Criteria (from spec)

1. **Milestone completion** — tracked via week-wise milestones above
2. **Accuracy of subject isolation** — IoU & Dice on val set, visual inspection of grids
3. **Documentation & presentation** — this README + `outputs/predictions/eval_grids/`

---

## References

- [U-Net paper](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [COCO dataset](https://cocodataset.org/)
- [Albumentations](https://albumentations.ai/)
