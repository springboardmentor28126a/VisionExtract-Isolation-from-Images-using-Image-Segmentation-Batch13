# VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13

## Project Overview
VisionExtract is an image segmentation project aimed at automatically isolating the main subject from an image. Given an input image, the system produces an output where the subject is retained while the background is completely blacked out. This functionality is useful in photography automation, digital art, background replacement, augmented reality, and virtual conferencing.

---

## Project Objectives
- Implement semantic image segmentation for subject isolation
- Preprocess image and mask data for pixel-wise learning
- Train and fine-tune a deep learning segmentation model
- Evaluate performance using standard segmentation metrics
- Validate model generalization on unseen data

---

## Dataset
- Dataset: COCO 2017 (val2017 subset)
- Total Images Used: 5000
- Annotations: Pixel-wise segmentation masks

---

# Milestone 1: Dataset Handling and Preprocessing

## Week 1: Project Initialization and Dataset Acquisition
- Defined project objectives and expected results
- Downloaded and set up the COCO 2017 dataset
- Explored dataset structure (images, annotations, categories)
- Visualized sample images and their segmentation masks
- Verified subject isolation using mask overlays

**Outcome:** Clear understanding of dataset structure and subject-mask relationships.

---

## Week 2: Data Preprocessing and Validation
- Implemented preprocessing pipeline:
  - Image resizing (256×256)
  - Image normalization (0–1)
  - Mask resizing using nearest-neighbor interpolation
- Converted multi-class masks into binary subject-background masks
- Applied data augmentation:
  - Horizontal flipping
  - Rotation
  - Zooming
  - Brightness adjustment
  - Gaussian blur
- Validated alignment between images and masks using visual inspection

**Outcome:** Robust preprocessing pipeline ready for model training.

---

# Milestone 2: Model Training, Evaluation, and Fine-Tuning

## Week 3: Initial Model Training
- Implemented U-Net based image segmentation model
- Used all 5000 images from COCO val2017
- Dataset split:
  - 70% Training
  - 15% Validation
  - 15% Testing
- Trained model using Binary Cross-Entropy loss
- Monitored training and validation loss
- Visualized early predictions to verify learning

**Outcome:** Functional segmentation model capable of subject extraction.

---

## Week 4: Predictions and Fine-Tuning
- Generated predictions on validation data
- Compared predicted masks with ground truth masks
- Evaluated using:
  - Intersection over Union (IoU)
  - Dice Coefficient
  - Pixel-wise Accuracy
- Fine-tuned model using combined BCE + Dice loss
- Adjusted learning rate for better boundary segmentation
- Saved final fine-tuned model

**Outcome:** Improved segmentation accuracy and refined predictions.

---

## Test Set Evaluation
- Evaluated final model on unseen test dataset (15%)
- Computed IoU, Dice score, and Pixel-wise accuracy
- Visualized predictions to confirm generalization

---

## Technologies Used
- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- COCO API (pycocotools)
- Google Colab

---

## Project Status
- Milestone 1: Completed
- Milestone 2: Completed

---

## Author
Himanshu Ahirrao