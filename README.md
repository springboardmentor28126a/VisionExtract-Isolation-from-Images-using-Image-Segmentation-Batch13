# VisionExtract — Isolation from Images using Image Segmentation

## Milestone 1 — COCO Mask Extraction & Subject Isolation

This module implements the first stage of the VisionExtract project — automatically generating segmentation masks from the COCO 2017 dataset and isolating the subject by removing the background.

The system reads COCO annotations, builds a binary mask for all objects in an image, and produces a subject-isolated output where background pixels are set to black.

---

## Project Objective

To verify dataset setup and build a working preprocessing pipeline that:

* Loads COCO dataset annotations
* Reads images correctly
* Generates segmentation masks
* Applies masks to remove background
* Saves subject-isolated outputs

---

## Dataset Used

COCO 2017 Dataset

Required structure:

```
COCO_FINAL/
 ├── train2017/
 └── annotations/
      instances_train2017.json
```

Dataset is stored locally and not included in this repository due to size.

---

## Files in This Module

* `path_test.py` — verifies dataset folder and structure
* `segmentation_basic.py` — loads COCO annotations and generates masks
* `requirements.txt` — required Python libraries
* `outputs/` — generated sample outputs
* `LICENSE` — project license

---

## Processing Pipeline

1. Load COCO annotation file
2. Select image from dataset
3. Read image using OpenCV
4. Extract all object masks
5. Merge into binary mask
6. Apply mask to image
7. Background pixels set to black
8. Save outputs

---

## Output Files

Generated inside `outputs/`:

* original.png — original image
* mask.png — binary segmentation mask
* isolated.png — subject isolated image

---

## How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run:

```
python segmentation_basic.py
```

---

## Result

This milestone confirms:

* Dataset connectivity works
* Annotation parsing works
* Mask generation works
* Subject isolation works

Next milestone will focus on training a deep learning segmentation model.
