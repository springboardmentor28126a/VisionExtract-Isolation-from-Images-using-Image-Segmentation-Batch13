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