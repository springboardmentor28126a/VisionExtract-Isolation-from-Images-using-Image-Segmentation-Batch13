# VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13
# COCO Data Preprocessing Pipeline

## Overview

This module contains scripts for preprocessing images from the **COCO 2017 dataset** and generating corresponding **segmentation masks** using the provided annotation JSON files.

The pipeline performs:

* Image loading
* Image preprocessing (resize, denoise, contrast enhancement, normalization)
* COCO annotation parsing
* Binary mask generation
* Image–mask alignment for segmentation tasks

This data is prepared for **deep learning / computer vision segmentation models**.

---

## Folder Structure

```
Data_Preprocessing/
│
├── preprocessing.py              # Image preprocessing pipeline
├── preprocess_coco_mask.py       # Generate COCO masks + resize
├── img_viz.py                    # Visualization (original vs processed vs mask)
├── sample_image.py               # Test script for a single image
├── requirements.txt              # Required Python libraries
└── README.md
```

COCO Dataset Location (Example):

```
coco2017/
│
├── train2017/                    # Training images
├── val2017/                      # Validation images
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

---

## Dataset

Dataset Used: **COCO 2017**

Download from:
https://cocodataset.org/#download

Required files:

* `train2017.zip`
* `val2017.zip`
* `annotations_trainval2017.zip`

After extraction:

```
coco2017/
    train2017/
    val2017/
    annotations/
```

---

## Preprocessing Steps

### Image Preprocessing

Performed in `preprocessing.py`:

1. Read image
2. Convert BGR → RGB
3. Resize to 512 × 512
4. Gaussian Blur (noise reduction)
5. CLAHE (contrast enhancement)
6. Normalize pixel values to range [0,1]

Output:

* Model-ready image tensor

---

### Mask Generation

Performed in `preprocess_coco_mask.py`:

1. Load COCO annotations using `pycocotools`
2. Find image ID from filename
3. Extract all object segmentations
4. Merge into a single binary mask
5. Resize mask to **512 × 512**
6. Ensure mask is binary (0 = background, 1 = object)

Important:

* Mask resizing uses **INTER_NEAREST** to avoid distortion.
* Image and mask sizes are always aligned.

---

## Visualization

`img_viz.py` displays:

* Original image
* Preprocessed image
* Binary segmentation mask

This helps verify preprocessing and annotation correctness.

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Typical libraries:

```
opencv-python
matplotlib
numpy
pycocotools
```

---

## How to Run

### 1. Preprocess a sample image

```bash
python preprocessing.py
```

### 2. Generate mask from COCO annotations

```bash
python preprocess_coco_mask.py
```

### 3. Visualize results

```bash
python img_viz.py
```

---

## Output Format

| Data  | Shape         | Type          |
| ----- | ------------- | ------------- |
| Image | (512, 512, 3) | float32 (0–1) |
| Mask  | (512, 512)    | binary (0/1)  |

This format is suitable for:

* U-Net
* Mask R-CNN
* Custom segmentation models

---

## Notes

* Always apply the **same transformations** to image and mask.
* Do not use interpolation methods other than **nearest** for masks.
* Ensure COCO annotation paths are correctly set inside the scripts.

---

## Use Case

This preprocessing pipeline is designed for:

* Instance segmentation
* Semantic segmentation
* Dataset preparation for deep learning models
* Computer vision training workflows




