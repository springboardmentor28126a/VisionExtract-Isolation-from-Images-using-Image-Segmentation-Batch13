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




---------------------------------------------------------------------

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