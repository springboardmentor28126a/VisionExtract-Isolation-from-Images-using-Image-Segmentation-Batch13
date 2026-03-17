# VisionExtract: Subject Isolation from Images using Image Segmentation

## Project Overview
VisionExtract is an image segmentation project aimed at automatically isolating the main subject from an image. Given an input image, the system produces an output where the subject is retained while the background is completely blacked out. This functionality is useful in photography automation, digital art, background replacement, augmented reality, and virtual conferencing.

The project leverages deep learning-based semantic segmentation techniques to classify each pixel of an image as either belonging to the subject or the background. By learning pixel-level features from annotated datasets, the system can automatically detect objects and remove unwanted background regions.

---

## Project Objectives
- Implement semantic image segmentation for subject isolation
- Preprocess image and mask data for pixel-wise learning
- Train and fine-tune a deep learning segmentation model
- Evaluate performance using standard segmentation metrics
- Validate model generalization on unseen data
- Improve segmentation quality using post-processing techniques

---

## Dataset
- Dataset: COCO 2017 (val2017 subset)
- Total Images Used: 5000
- Annotations: Pixel-wise segmentation masks
- Source: https://cocodataset.org

The COCO dataset is a widely used dataset in computer vision research containing real-world images with detailed annotations. Each image contains object segmentation masks that define the exact boundaries of objects present in the image.

---

# Milestone 1: Dataset Handling and Preprocessing

## Week 1: Project Initialization and Dataset Acquisition
- Defined project objectives and expected results
- Downloaded and set up the COCO 2017 dataset
- Explored dataset structure (images, annotations, categories)
- Visualized sample images and their segmentation masks
- Verified subject isolation using mask overlays
- Inspected annotation JSON files and category labels

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
- Ensured consistent mask boundaries after resizing

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
- Used Adam optimizer for efficient gradient updates
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
- Compared results with initial training performance

Example Model Performance:

Metric | Initial Model | Fine-Tuned Model
------ | ------------- | ---------------
IoU | 0.4505 | 0.5502
Dice Score | 0.5837 | 0.6521
Accuracy | 0.7738 | 0.8125

---

# Milestone 3: Model Improvement and Inference

## Week 5: Model Enhancement
- Improved preprocessing pipeline
- Added additional augmentation techniques
- Tested model robustness on diverse images
- Refined training process for better feature learning

**Outcome:** Enhanced model stability and robustness.

---

## Week 6: Inference and Subject Isolation
- Implemented inference pipeline for new images
- Generated predicted segmentation masks
- Applied thresholding to convert probability masks into binary masks
- Multiplied binary mask with original image to isolate subject
- Produced final output images with background removed

Example inference workflow:

Input Image → Segmentation Model → Predicted Mask → Binary Mask → Subject Isolation

**Outcome:** Successfully isolated subjects from new unseen images.

---

# Milestone 4: Full Pipeline and User Interface

## Week 7: Pipeline Integration and Intermediate Results
- Built a basic web interface where users can upload an image
- Integrated the preprocessing, model inference, and post-processing into a single pipeline
- Implemented **display of intermediate results** for demo purposes:
  - `image` → Original image
  - `mask1` → Output from Mask R-CNN
  - `mask2` → Output from DeepLabV3
  - `final_mask` → Combined refined mask
  - `isolated` → Final subject isolated image
- Enabled step-by-step visualization for debugging and presentation

Example Pipeline:

Input Image → Preprocessing → Mask1 (Mask R-CNN) → Mask2 (DeepLabV3) → Final Mask → Isolated Subject → Output Display

**Outcome:** Full working pipeline with interactive demo ready for presentations.

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
- Milestone 3: Completed
- Milestone 4: Completed

---

## Future Improvements
- Train model on larger datasets
- Implement advanced segmentation architectures such as DeepLabV3+
- Improve boundary detection using attention mechanisms
- Deploy the system as a web application with interactive interface

---

## Author
Himanshu Ahirrao