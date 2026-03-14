#VisionExtract
#Binary Semantic Segmentation for Background Removal
## Overview

VisionExtract is a deep learning-based computer vision project that performs binary semantic segmentation to isolate the main subject (Person) from an image and remove the background.

The system uses the COCO val2017 dataset and converts polygon annotations into binary masks for supervised training.

##Project Goal

The objective of this project is to:

Take an RGB image as input

Identify the main subject (Person category)

Generate a binary segmentation mask

Enable background removal

Project Type

This project belongs to:

Computer Vision

Supervised Learning

Binary Semantic Segmentation

Deep Learning (CNN-based approach)

Dataset Used

The project uses:

COCO val2017 Dataset

Dataset Components:

val2017/ → 5000 images

instances_val2017.json → Segmentation annotations

Only the Person category is used to convert the task into binary segmentation.

 Dataset Processing

COCO provides segmentation data as polygon annotations, not mask images.

Steps performed:

Loaded annotation JSON file

Filtered images containing the Person category

Converted polygon annotations into binary masks using pycocotools

Verified mask correctness through visualization

Binary mask format:

1 → Person

0 → Background

Data Preprocessing

To prepare data for training, the following preprocessing steps were implemented:

✔ Image Resizing

All images and masks were resized to:

256 × 256


This ensures consistent input size for neural networks.

✔ Normalization

Images were normalized using ImageNet mean and standard deviation to improve training stability and compatibility with pretrained encoders.

✔ Binary Conversion

Multi-class segmentation annotations were simplified into binary segmentation:

Person pixels → 1

All other pixels → 0

✔ Dataset Splitting

The dataset was shuffled and split into:

70% → Training

15% → Validation

15% → Testing

This ensures proper model evaluation and prevents data leakage.

 Data Pipeline

The project includes:

Custom PyTorch Dataset class

DataLoader pipeline

Mask generation function

Preprocessing transformations

Structured train/validation/test splits

The dataset is fully model-ready.

 Technologies Used

Python

PyTorch

Torchvision

NumPy

Matplotlib

pycocotools

Current Status

✔ Dataset acquisition completed
✔ Polygon-to-mask conversion implemented
✔ Preprocessing pipeline created
✔ Dataset splitting completed
✔ DataLoader ready

Next step:

Implement U-Net model

Train segmentation model

Evaluate using IoU and Dice metrics

 Key Learnings

Understanding COCO annotation structure

Converting polygon annotations to masks

Building a clean preprocessing pipeline

Handling binary segmentation tasks

Preventing data leakage through proper splitting

🏁 Conclusion

VisionExtract establishes a structured data engineering pipeline for binary semantic segmentation. The dataset is prepared and validated for deep learning-based training and evaluation.
