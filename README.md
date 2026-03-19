# VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13

**************************************************
# VisionExtract – Milestone 1
Subject Isolation using Image Segmentation
# Project Overview

VisionExtract is a deep learning-based image segmentation project aimed at automatically isolating the main subject from an image. The output image retains only the subject while the background is rendered completely black.

This milestone focuses on dataset acquisition, exploration, and preprocessing pipeline development.

# Milestone 1 Objectives

Acquire and inspect the COCO 2017 dataset
Extract images containing the "person" category
Generate binary segmentation masks
Build a preprocessing pipeline
Validate image-mask alignment
Prepare dataset for training

# Dataset Used

-Dataset Name: COCO 2017
-Source: Kaggle
-Category Used: Person
-Total Images Identified: 64,115
-The dataset provides:
Images (train2017)
Annotation file (instances_train2017.json)
Polygon segmentation masks

# Technologies Used

-Python
-TensorFlow
-NumPy
-Matplotlib
-PIL
-pycocotools

# Dataset Exploration

Retrieved category ID for "person"
Extracted image IDs containing that category
Loaded sample images and corresponding masks
Visualized original images alongside generated binary masks

Mask generation process:
Converted polygon annotations into binary masks
Merged multiple person instances into a single mask
Background = 0
Subject (Person) = 1

# Preprocessing Pipeline

A TensorFlow tf.data pipeline was implemented for efficient data handling.
Steps Performed:
Load image and generate mask using COCO API
Resize images and masks to 256×256
Normalize images (0–255 → 0–1)
Convert masks to binary format
Batch data (batch size = 16)
Prefetch for optimized performance
Output Validation

Pipeline successfully generated:
Batch Image Shape: (16, 256, 256, 3)
Batch Mask Shape: (16, 256, 256, 1)

This confirms:
Correct resizing
Proper alignment
Successful batching
Mask channel correctness

# Milestone 1 Achievements

✔ Dataset acquisition and validation
✔ Category-specific image filtering
✔ Binary mask generation
✔ Efficient tf.data pipeline implementation
✔ Shape validation and alignment verification




-------------------------------------------------------------------------------------------------------------------------

# Milestone 2 Objectives
Split the dataset into formal training, validation, and testing sets

Build a memory-efficient custom data generator for batch processing

Implement a U-Net architecture (Encoder/Decoder network)

Execute baseline model training and monitor loss metrics

Evaluate predictions mathematically using Intersection over Union (IoU)

Implement data augmentation and fine-tune the model for higher accuracy

Visualize and validate final predicted masks against ground truth

# Dataset Used
Subset Used: 5,000 images from COCO val2017

Split Strategy:

70% Training (3,500 images)

15% Validation (750 images)

15% Testing (750 images)

Image/Mask Dimensions: Resized to 128×128

Normalization: RGB values scaled from 0–255 to 0–1

# Technologies Used
Python

TensorFlow / Keras

OpenCV (cv2)

NumPy

Matplotlib

pycocotools

# Model Architecture (U-Net)
A lightweight U-Net architecture was built from scratch using tf.keras:

Encoder (Downsampling): Utilizes Conv2D and MaxPooling2D layers to shrink the image and extract core features (learning what the subject is).

Bottleneck: The deepest layer holding the most compressed feature representations.

Decoder (Upsampling): Utilizes UpSampling2D and skip connections (layers.concatenate) to reconstruct the spatial dimensions (learning where the subject is).

Output Layer: A Conv2D layer with a Sigmoid activation function to generate a 1-channel probability map for the binary mask.

Training Pipeline & Fine-Tuning
A custom tf.keras.utils.Sequence generator was implemented to load images from the hard drive in batches of 16, preventing RAM overflow.

# Phase 1: Baseline Training (Week 3)

Optimizer: Adam (Standard learning rate)

Loss Function: Binary Cross-Entropy

Epochs: 5

Result: Training and Validation loss decreased simultaneously, confirming healthy learning without overfitting.

# Phase 2: Fine-Tuning (Week 4)

Data Augmentation: Added RandomFlip and RandomRotation (10%) to the training pipeline to make the model robust to different orientations.

Optimizer: Adam (Lowered learning rate to 1e-4)

Epochs: 10

Result: Locked in optimized weights and saved the final model as final_segmentation_model.keras.

# Output Validation
Pipeline successfully verified model intelligence through two methods:

Visual Verification: Matplotlib scripts successfully generated side-by-side comparisons of the Original RGB Image, the True Ground-Truth Mask, and the Model's Predicted Mask.

Mathematical Verification: Implemented a custom Intersection over Union (IoU) function to mathematically score the exact pixel overlap between predictions and true masks.





------------------------------------------------------------------------------------------------------------------------




# 🚀 Milestone 3 – Advanced Architecture & Automated Inference

## 📌 Overview

Milestone 3 focuses on transforming the VisionExtract system from a basic segmentation model into a **robust, optimized, and automated inference pipeline**. 

In this phase, we improved model performance using **transfer learning**, applied **advanced training strategies**, handled real-world edge cases, and built a complete **end-to-end subject isolation system**.

---

# 🧠 Objectives

- Improve segmentation accuracy
- Reduce overfitting during training
- Experiment with advanced architectures
- Build an automated inference pipeline
- Test model on real-world unseen images

---

# 🔧 Key Technical Improvements

## 1️⃣ Transfer Learning (MobileNetV2 + U-Net)

- Replaced traditional U-Net encoder with **MobileNetV2**
- Loaded **pre-trained weights from ImageNet**
- Frozen encoder layers to retain learned features

### ✅ Impact:
- Faster convergence
- Better feature extraction
- Improved segmentation performance

---

## 2️⃣ Advanced Training Strategies

Implemented TensorFlow callbacks:

### 🔹 EarlyStopping
- Monitors validation loss
- Stops training if no improvement for 3 epochs
- Restores best model weights automatically

### 🔹 ModelCheckpoint
- Saves best-performing model during training
- Prevents loss of optimal weights

### ✅ Impact:
- Prevents overfitting
- Reduces training time
- Ensures best model is always saved

---

## 3️⃣ Data Augmentation

Applied transformations:

- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation

### ✅ Impact:
- Improves generalization
- Makes model robust to variations
- Reduces overfitting

---

## 4️⃣ Model Comparison

Compared performance of:

- Basic U-Net (Milestone 2)
- MobileNet-based U-Net (Milestone 3)

### Evaluation Metrics:
- Validation Loss
- IoU (Intersection over Union)
- Visual Output Quality

---

# 🔬 Edge Case Analysis

During testing on real-world images, several limitations were observed.

## ❗ Observed Issues:
- Poor segmentation for:
  - Dark clothing
  - Hair edges
  - Low contrast backgrounds

---

## 🔍 Root Cause

### 1️⃣ Resolution Limitation
- Model trained on **128×128 images**
- Loss of fine details (hair, edges)

### 2️⃣ Dataset Bias
- COCO dataset contains mostly:
  - Wide-angle environmental images
- Lacks:
  - Close-up portraits
  - Studio lighting conditions

---

# 🛠️ Post-Processing Pipeline

To improve results without retraining, a custom OpenCV pipeline was implemented.

---

## 🔹 Step 1: Threshold Adjustment

- Lowered threshold from **0.5 → 0.2**

### Purpose:
- Capture low-confidence pixels (hair, shadows)

---

## 🔹 Step 2: Contour Filling

Used:
- `cv2.findContours`
- `cv2.drawContours`

### Purpose:
- Fill holes inside subject
- Remove segmentation gaps

---

## 🔹 Step 3: Median Blurring

Used:
- `cv2.medianBlur`

### Purpose:
- Smooth jagged edges
- Avoid halo effect

---

## ✅ Result:
- Cleaner masks
- Better edge continuity
- Improved visual output

---

# ⚡ Final Inference Pipeline

## 🔁 End-to-End Workflow

# 🚀 Milestone 3 – Advanced Architecture & Automated Inference

## 📌 Overview

Milestone 3 focuses on transforming the VisionExtract system from a basic segmentation model into a **robust, optimized, and automated inference pipeline**. 

In this phase, we improved model performance using **transfer learning**, applied **advanced training strategies**, handled real-world edge cases, and built a complete **end-to-end subject isolation system**.

---

# 🧠 Objectives

- Improve segmentation accuracy
- Reduce overfitting during training
- Experiment with advanced architectures
- Build an automated inference pipeline
- Test model on real-world unseen images

---

# 🔧 Key Technical Improvements

## 1️⃣ Transfer Learning (MobileNetV2 + U-Net)

- Replaced traditional U-Net encoder with **MobileNetV2**
- Loaded **pre-trained weights from ImageNet**
- Frozen encoder layers to retain learned features

### ✅ Impact:
- Faster convergence
- Better feature extraction
- Improved segmentation performance

---

## 2️⃣ Advanced Training Strategies

Implemented TensorFlow callbacks:

### 🔹 EarlyStopping
- Monitors validation loss
- Stops training if no improvement for 3 epochs
- Restores best model weights automatically

### 🔹 ModelCheckpoint
- Saves best-performing model during training
- Prevents loss of optimal weights

### ✅ Impact:
- Prevents overfitting
- Reduces training time
- Ensures best model is always saved

---

## 3️⃣ Data Augmentation

Applied transformations:

- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation

### ✅ Impact:
- Improves generalization
- Makes model robust to variations
- Reduces overfitting

---

## 4️⃣ Model Comparison

Compared performance of:

- Basic U-Net (Milestone 2)
- MobileNet-based U-Net (Milestone 3)

### Evaluation Metrics:
- Validation Loss
- IoU (Intersection over Union)
- Visual Output Quality

---

# 🔬 Edge Case Analysis

During testing on real-world images, several limitations were observed.

## ❗ Observed Issues:
- Poor segmentation for:
  - Dark clothing
  - Hair edges
  - Low contrast backgrounds

---

## 🔍 Root Cause

### 1️⃣ Resolution Limitation
- Model trained on **128×128 images**
- Loss of fine details (hair, edges)

### 2️⃣ Dataset Bias
- COCO dataset contains mostly:
  - Wide-angle environmental images
- Lacks:
  - Close-up portraits
  - Studio lighting conditions

---

# 🛠️ Post-Processing Pipeline

To improve results without retraining, a custom OpenCV pipeline was implemented.

---

## 🔹 Step 1: Threshold Adjustment

- Lowered threshold from **0.5 → 0.2**

### Purpose:
- Capture low-confidence pixels (hair, shadows)

---

## 🔹 Step 2: Contour Filling

Used:
- `cv2.findContours`
- `cv2.drawContours`

### Purpose:
- Fill holes inside subject
- Remove segmentation gaps

---

## 🔹 Step 3: Median Blurring

Used:
- `cv2.medianBlur`

### Purpose:
- Smooth jagged edges
- Avoid halo effect

---

## ✅ Result:
- Cleaner masks
- Better edge continuity
- Improved visual output

---

# ⚡ Final Inference Pipeline

## 🔁 End-to-End Workflow

Input Image
↓
Resize (128×128)
↓
Model Prediction (Mask)
↓
Thresholding
↓
Resize to Original Size
↓
Post-Processing
↓
Apply Mask
↓
Save Final Output



---

## 📸 Output

- Original Image  
- Binary Mask  
- Final Subject-Isolated Image  

---

# 📊 Evaluation Metrics

- **IoU (Intersection over Union)**
- Visual Inspection (Qualitative)

---

# 📁 Model Output

Saved Model:


---

# 🔮 Future Scope

To make this production-ready:

### 🔹 1. Higher Resolution Training
- Upgrade from:
  - 128×128 → 256×256 or 512×512

---

### 🔹 2. Specialized Datasets
- Use portrait/matting datasets:
  - MODNet
  - Human Matting datasets

---

### 🔹 3. Alpha Matting

Instead of binary masks:
- Use **soft masks (alpha blending)**

Better for:
- Hair
- Transparent edges

---

### 🔹 4. Advanced Models

- U²-Net
- DeepLabV3+
- Attention U-Net

---

# 👨‍💻 Author

**Sai Jannawar**  
CSE (Artificial Intelligence)

---

# 🎯 Conclusion

VisionExtract successfully demonstrates:

- End-to-end segmentation pipeline  
- Real-world problem solving  
- Model improvement using transfer learning  
- Practical deployment using inference pipeline  

The project evolves from a basic academic model into a **real-world AI system capable of subject isolation**.
