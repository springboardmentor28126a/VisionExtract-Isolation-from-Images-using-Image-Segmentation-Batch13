# VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13
# Contributions by Hithesh N


# 🚀 VisionExtract
## Subject Isolation Using Image Segmentation

### 📌 Project Overview

VisionExtract is a deep learning-based computer vision project focused on isolating the primary subject in an image using binary semantic segmentation.

The system learns to:

- Identify foreground objects
- Separate them from background
- Generate a binary mask
- Enable background removal

### 📅 Project Milestones

- ✅ Week 1 – Dataset Understanding & Mask Generation
- ✅ Week 2 – Data Preprocessing Pipeline
- 🔜 Week 3 – Model Architecture & Training

### 📂 Project Structure

```
VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13/
│
├── data/
│   ├── val2017/                  # COCO images
│   ├── annotations/              # COCO JSON annotations
│   └── splits/                   # Train/Val/Test split files
│
└── codes/
    ├── split_dataset.py
    ├── generate_mask.py
    ├── preprocessing.py
    ├── test_preprocessing.py
    └── visualize.py
```

---

## 🧠 Week 1 – Dataset Understanding & Preparation

### 📦 Dataset Used

We used the **COCO 2017 Dataset** (Common Objects in Context).

#### Why COCO?

- Real-world scene images
- Pixel-level segmentation annotations
- 80 object categories
- Widely used research benchmark
- Supports instance segmentation

### 📑 COCO Annotation Structure

The file: `instances_val2017.json`

Contains:

- **images** → image metadata
- **annotations** → polygon segmentation data
- **categories** → class label mapping

Example structure:

```json
{
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
```

### 🔀 Dataset Reduction Strategy

To reduce computational requirements:

- Used val2017 subset (~5000 images)
- Shuffled dataset
- Split into:

| Split | Percentage |
|-------|-----------|
| Train | 70%       |
| Val   | 15%       |
| Test  | 15%       |

**Why Shuffle?**

- Prevents ordering bias
- Ensures fair distribution
- Avoids data leakage

### 🎭 Mask Generation Strategy

COCO provides polygon-based instance segmentation.

We converted polygons into pixel masks using:

- `pycocotools`

#### Design Decision

Instead of selecting a single class (e.g., person), we merged all 80 classes into one foreground category.

This converts:

**Multi-class instance segmentation → Binary semantic segmentation**

#### Binary representation:

```python
if pixel > 0:
    pixel = 1
else:
    pixel = 0
```

- **Foreground** → 1
- **Background** → 0

### 📘 Important Concepts (Week 1)

#### 🔹 Semantic Segmentation

Assigns a class label to every pixel in an image.

#### 🔹 Instance Segmentation

Separates individual object instances.

- COCO supports instance segmentation.
- This project reformulates it into binary semantic segmentation.

#### 🔹 Ground Truth

Annotated masks provided in dataset used for training and evaluation.

#### 🔹 Lazy Mask Generation

Masks are generated dynamically during training instead of being saved to disk.

**Benefits:**

- Saves storage
- Efficient memory usage
- Scalable design

---

## 🧪 Week 2 – Data Preprocessing Pipeline

### 🎯 Objective

Develop a preprocessing pipeline that:

- Resizes images & masks
- Normalizes pixel values
- Applies safe augmentation
- Maintains spatial alignment
- Converts masks to binary

### 📏 Resizing

All images resized to: **256 x 256**

**Why?**

- Required for batch training
- Improves GPU efficiency
- Standardizes input size

#### Mask Interpolation Method: `INTER_NEAREST`

**Why?**

- Prevents fractional label values
- Preserves class boundaries

### 🎨 Normalization

```python
image = image / 255.0
```

Converts pixel range: **[0, 255] → [0, 1]**

**Benefits:**

- Stable gradient updates
- Faster convergence
- Better optimization

### 🔄 Data Augmentation

Initially tested:

- Random rotation
- Horizontal flip
- Brightness adjustment

#### Rotation Removed ❌

**Reason:**

- COCO contains natural scenes
- Buildings and people follow gravity
- Strong rotations created unrealistic orientations

#### Final augmentation strategy:

- Horizontal flip
- Brightness variation

### 🔗 Spatial Alignment (Critical Concept)

In segmentation tasks:

> **If image is transformed, mask must undergo the exact same transformation.**

Otherwise:

- Pixel mismatch occurs
- Model learns incorrect mapping
- Training becomes unstable

This principle ensures pixel-level correspondence.

### 🏗 Training vs Inference Pipeline

Two separate preprocessing pipelines were created:

#### 🔹 Training Pipeline

- Augmentation enabled
- Requires image + mask

#### 🔹 Inference Pipeline

- No augmentation
- Deterministic
- Only resize + normalize

**Why separate?**

Training requires randomness. Inference must remain stable and predictable.

---

## 🧠 Research-Based Design Decisions

### Why Binary Segmentation?

- Simplifies optimization
- Reduces class imbalance
- Matches subject isolation goal

### Why Dynamic Mask Generation?

- Avoids storing thousands of mask images
- More scalable
- Faster experimentation

### Why 256x256 Resolution?

- Balances detail preservation
- Reduces memory usage
- Suitable for mid-range GPUs

---

## 📊 Current Status

- ✔ Dataset understanding complete
- ✔ Dataset split implemented
- ✔ Dynamic mask generation working
- ✔ Binary conversion validated
- ✔ Preprocessing pipeline implemented
- ✔ Augmentation strategy refined

---

## 🔜 Next Phase – Week 3

- U-Net architecture
- Custom Dataset class
- Loss function (BCE + Dice)
- IoU metric
- Training loop
- GPU optimization

---

## 🎯 Conclusion

Weeks 1 and 2 established the data foundation of the project.

The system is now fully prepared for model development and training.


