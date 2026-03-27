# VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13
# COCO Data Preprocessing Pipeline

🧠 Subject Segmentation using COCO Dataset
📌 Project Overview

This project focuses on automatic subject segmentation using the COCO 2017 validation dataset. The goal is to extract the main object (foreground) from images using deep learning.

The pipeline includes:

Image preprocessing
Binary mask generation (largest object selection)
Dataset creation
Deep learning model training (DeepLabV3+)
Deployment via Streamlit web application
📂 Dataset Used
Dataset: COCO 2017
Split: Validation set (val2017)
Total images used: 5000
Annotation file: instances_val2017.json



⚙️ Preprocessing Pipeline

Each image undergoes the following steps:

Resize
All images resized to 512 × 512
Denoising
Gaussian Blur applied to remove noise
Contrast Enhancement
CLAHE applied on L-channel (LAB color space)
Normalization
Pixel values scaled to [0, 1]
Final Output
Clean, enhanced image suitable for training



🎯 Mask Generation Strategy

Using COCO annotations:

Extract all object masks per image
Select largest object (main subject)
Convert to binary mask
Resize to 512 × 512
Apply filtering:
❌ Skip images with no annotations
❌ Skip very small objects
❌ Skip low foreground ratio

Apply morphological operations:
Closing (fill gaps)
Opening (remove noise)

Generate:
🟢 Binary mask
🟢 Masked image (background removed)



📌 Implementation:

🗂️ Output Dataset Structure
processed_binary/
│
├── images/        # Masked images (foreground only)
├── masks/         # Binary masks

🧩 Model Architecture
Model: DeepLabV3+
Encoder: ResNet50 (ImageNet pretrained)
Framework: segmentation_models_pytorch



🏋️ Training Details
Input size: 256 × 256
Batch size: 8
Epochs: 30
Optimizer: Adam
Learning rate: 1e-4
Loss Function

Combined loss:

Binary Cross Entropy (BCE)
Dice Loss
Metrics
IoU (Intersection over Union)
Pixel Accuracy



📌 Training code:

📊 Dataset Split
Training: 70%
Validation: 15%
Testing: 15%



🚀 Model Performance
Best model saved based on Validation IoU
Final evaluation performed on test set


💻 Streamlit Web Application

An interactive UI is built to:

Upload image
Generate segmentation mask
Extract main subject
Download result
Features:
Clean UI with animations
Real-time inference
Side-by-side comparison


🔄 Inference Pipeline
Upload image
Resize → Normalize
Model prediction
Threshold mask
Extract subject


🧪 Sample Output
Original Image
Binary Mask
Extracted Subject