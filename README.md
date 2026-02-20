# VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13

## 📌 Milestone 1 – Project Initialization & Data Preparation

### 🔍 Project Objective
The goal of this project is to build a deep learning model capable of automatically isolating selected subjects (person, animals, fruits) from an image using semantic segmentation. The output image retains only the detected subject while rendering the background completely black.

---

## ✅ Milestone 1 Completed Tasks

### 1️⃣ Dataset Acquisition
- Downloaded **COCO 2017 Dataset**
- Loaded annotations using `pycocotools`
- Explored dataset structure (images + segmentation masks)

### 2️⃣ Category Selection
Selected the following object categories for subject isolation:
- **Person**
- **Animals** (dog, cat, horse, sheep, cow, elephant, etc.)
- **Fruits / Food Items** (banana, apple, orange, pizza, etc.)

### 3️⃣ Mask Generation
- Extracted segmentation annotations from COCO
- Converted multi-class masks into **binary masks**
    - `1` → Selected object
    - `0` → Background
- Ensured correct image–mask alignment

### 4️⃣ Data Preprocessing
- Resized images and masks to **256×256**
- Normalized image pixel values (0–1 range)
- Converted masks to single-channel binary format
- Implemented custom PyTorch Dataset class

### 5️⃣ DataLoader Setup
- Created training and validation splits
- Implemented PyTorch DataLoader for batch processing

---

## 🛠 Tools & Technologies Used
- Python 3.x
- Google Colab (GPU enabled)
- PyTorch
- Segmentation Models PyTorch
- COCO Dataset
- pycocotools
- OpenCV
- NumPy
- Matplotlib

---

## 📊 Sample Binary Mask Output
<img width="1320" height="1190" alt="image" src="https://github.com/user-attachments/assets/17b1912e-d1a5-4233-a332-fdad36721008" />
<img width="1320" height="1190" alt="image" src="https://github.com/user-attachments/assets/17b1912e-d1a5-4233-a332-fdad36721008" />

---

### 👨‍💻 Author
Rahul Raj 
