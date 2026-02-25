# VisionExtract: Subject Isolation from Images using Image Segmentation

VisionExtract is a computer vision project designed to automatically extract the main subject from an image using segmentation techniques. The system generates binary masks from COCO segmentation annotations and isolates the subject by removing background pixels.

---

## 📂 Project Structure

VisionExtract/
│
├── data/
│   ├── images/val2017
│   ├── annotations/
│   ├── masks/
│
├── src/
│   ├── dataset.py
│
├── notebooks/
│
├── requirements.txt
└── README.md

---

## ⚙️ Environment Setup

- Python 3.10
- PyTorch
- torchvision
- OpenCV
- NumPy
- Matplotlib
- pycocotools

Install dependencies:

pip install -r requirements.txt

---

## 📊 Dataset

Dataset used: COCO 2017 Validation Set

Includes:
- 5000 images
- Segmentation annotations
- 80 object categories

---

## 🔄 Phase 3: Mask Generation

- Load COCO annotation JSON
- Map category IDs to names
- Extract segmentation polygons
- Generate binary mask
- Apply mask to isolate subject
- Save mask and isolated image

---

## 🔧 Phase 4: Data Preprocessing

- Automate mask generation for multiple images
- Resize images and masks
- Normalize pixel values (0–1)
- Convert to PyTorch tensors
- Split dataset into training and validation sets

---

## 📈 Next Steps

- Implement segmentation model (U-Net)
- Train using image-mask pairs
- Evaluate using validation metrics
- Optimize performance

---

## 🎯 Objective

To build a model capable of isolating the primary subject in an image by learning pixel-level segmentation.
