#  VisionExtract

## Subject Isolation from Images using Deep Learning Segmentation

VisionExtract is a **computer vision deep learning project** that performs **binary semantic segmentation** to isolate the main subject (person) from an image.

The system processes an input image and generates a **binary mask** to separate the subject from the background.

The project follows a structured deep learning workflow across **three milestones:**

* Dataset Preparation
* Model Training & Evaluation
* Model Improvement & Inference Deployment

---

#  Project Objective

The main objective is to build an AI system that:

* Takes an RGB image as input
* Identifies the main subject (person)
* Generates a binary segmentation mask
* Removes background automatically
* Produces subject-isolated output images

---

#  Project Pipeline

```text
COCO Dataset + Annotations
        ↓
Polygon → Binary Mask Conversion
        ↓
Preprocessing Pipeline
        ↓
Train / Validation / Test Split
        ↓
U-Net Segmentation Training
        ↓
Evaluation using IoU Metric
        ↓
Model Improvement & Architecture Experiments
        ↓
Inference Pipeline for New Images
        ↓
Subject Extraction Output
```

---

#  Tech Stack

* Python
* PyTorch
* segmentation-models-pytorch
* NumPy
* OpenCV
* Matplotlib
* COCO Dataset

---

#  Dataset

The project uses a subset of the **COCO val2017 dataset**.

Dataset structure:

```text
val2017/
   ├── image1.jpg
   ├── image2.jpg
   └── ...

annotations/
   └── instances_val2017.json
```

COCO provides **polygon annotations**, which were converted into **binary masks**:

* Person → 1
* Background → 0

---

#  Milestone 1 – Dataset Preparation

Milestone 1 focused on building a **clean and structured data pipeline** for segmentation training.

### Tasks Completed

* Defined segmentation problem and expected output
* Downloaded COCO dataset
* Filtered Person category for binary segmentation
* Converted polygon annotations into pixel-level binary masks
* Built preprocessing pipeline (resize, normalize, tensor conversion)
* Verified image-mask alignment visually
* Split dataset into Train / Validation / Test (70 / 15 / 15)
* Implemented custom Dataset and DataLoader classes

### Outcomes

* Model-ready segmentation dataset created
* Clean binary masks generated
* Efficient batch data loading pipeline established

---

#  Milestone 2 – Model Training & Evaluation

Milestone 2 focused on implementing and training the segmentation model.

---

##  Model Architecture

The project uses **U-Net**, a CNN designed specifically for image segmentation.

Architecture features:

* Encoder-decoder structure
* Skip connections to preserve spatial information
* Pixel-wise mask prediction

Encoder used:

**ResNet34 pretrained on ImageNet**

Benefits:

* Faster learning
* Better feature extraction
* Improved segmentation quality

---

##  Loss Function

Binary segmentation requires pixel-wise comparison.

Loss used:

**Binary Cross Entropy with Logits Loss**

This measures the difference between predicted masks and ground truth masks.

---

##  Optimizer

**Adam Optimizer**

Advantages:

* Adaptive learning rate
* Stable convergence
* Effective gradient updates

---

##  Model Evaluation

Segmentation performance measured using **Intersection over Union (IoU)**.

Formula:

```text
IoU = Intersection / Union
```

Where:

* Intersection → overlap between predicted and true mask
* Union → total combined area

Higher IoU indicates better segmentation.

---

##  Prediction Visualization

Model predictions were visually compared with ground truth masks:

* Original image
* Ground truth mask
* Predicted mask

This qualitative validation ensured proper subject extraction.

---

##  Hyperparameter Tuning

Model performance improved by adjusting:

* Learning rate
* Batch size
* Number of epochs

---

##  Data Augmentation

Applied:

* Random horizontal flip
* Random rotation

Benefits:

* Prevents overfitting
* Improves generalization
* Helps model handle varied subject orientations

---

## Outcomes of Milestone 2

* Working segmentation model trained
* Loss monitored and reduced
* IoU metric calculated
* Visual segmentation validation completed
* Performance improved through tuning

---

#  Milestone 3 – Model Improvement & Inference Deployment

Milestone 3 focused on **enhancing segmentation quality and deploying the trained model for real-world inference.**

---

##  Improved Data Processing

Advanced augmentation techniques added:

* Vertical flip
* Color jitter
* Stronger rotation

This increased dataset diversity and improved model robustness.

---

##  Mask Post-Processing

Morphological operations applied to predicted masks:

* Closing → fills holes
* Opening → removes noise

Result:

Cleaner subject boundaries and improved visual output.

---

##  Architecture Experimentation

Multiple segmentation architectures were tested:

* U-Net + ResNet34
* U-Net + ResNet50
* DeepLabV3+

Performance comparison conducted based on:

* IoU score
* Training speed
* Mask sharpness

This experimentation helped identify the best performing model.

---

##  Inference Pipeline

A complete inference system was developed to process **new unseen images.**

Pipeline steps:

1. Load trained model weights
2. Apply same preprocessing
3. Predict segmentation mask
4. Apply thresholding
5. Extract subject by removing background
6. Save final output image

This enables automated subject isolation.

---

##  Robustness Testing

Model tested on:

* Internet images
* Personal photos
* Different lighting conditions
* Different poses

This validated model generalization beyond training data.

---

## Outcomes of Milestone 3

* Improved segmentation accuracy
* Cleaner masks through post-processing
* Multiple architectures evaluated
* Automated inference pipeline built
* Subject extraction working on unseen images

---

#  Final Result

The VisionExtract system successfully performs:

* Binary semantic segmentation
* Subject isolation
* Background removal
* Automated inference on new images

This project demonstrates a **complete deep learning lifecycle:**

Data Engineering → Model Training → Model Optimization → Deployment Pipeline

---

#  Future Improvements

* Multi-class segmentation
* Higher resolution training
* Real-time video segmentation
* Web app deployment
* Mobile inference optimization

---
# VisionExtract — Milestone 4: Full Pipeline and User Interface

## Overview

Milestone 4 focuses on integrating the complete subject extraction pipeline into a user-friendly application and preparing the project for final demonstration and presentation. This stage transforms the trained model into a usable system capable of processing real-world inputs.

---

## Week 7: Full Pipeline and User Interface

### Objective

To build a complete end-to-end system where users can upload an image and receive a subject-isolated output using the trained segmentation model.

---

## System Pipeline Integration

The full pipeline integrates preprocessing, model inference, and output generation into a single streamlined workflow.

### Pipeline Steps

1. **Input Image Upload**
   - User uploads an image through the interface.

2. **Preprocessing**
   - Image is resized to model input size (320×320).
   - Normalization is applied using ImageNet standards.

3. **Model Inference**
   - Preprocessed image is passed to the trained DeepLabV3+ model.
   - Model generates a probability mask.

4. **Mask Processing**
   - Thresholding is applied to convert probability mask into binary mask.
   - Largest connected component is selected as the main subject.
   - Mask is refined using smoothing and morphological operations.

5. **Resolution Restoration**
   - Processed mask is resized back to the original image resolution.

6. **Output Generation**
   - Background is removed using the mask.
   - Output is generated as:
     - Black background image OR
     - Transparent PNG image (optional)

---

## User Interface Implementation

### Technology Used

- Gradio (Python-based web UI framework)

### Features

- Image upload interface
- Adjustable threshold slider for segmentation control
- Edge smoothing control
- Transparent background output option
- Real-time result visualization

---

## Sample Interface Workflow

1. Upload an image
2. Adjust threshold and smoothing parameters
3. Enable/disable transparent background
4. Click submit to process
5. View extracted subject output

---

## Key Functional Components

### Inference Function

- Handles preprocessing, prediction, and post-processing
- Ensures output matches original image resolution

### Mask Refinement

- Applies smoothing for edge quality
- Uses connected component analysis to isolate main subject

### Output Rendering

- Supports RGB output (black background)
- Supports RGBA output (transparent background)

---

## Week 8: Documentation, Presentation, and Demo

### Documentation

- Compiled complete project documentation including:
  - Data preprocessing pipeline
  - Model architecture and training process
  - Inference workflow
  - Post-processing techniques

---

### Presentation Preparation

- Created presentation covering:
  - Problem statement and objectives
  - Methodology and pipeline design
  - Model training and evaluation
  - System architecture
  - Results and improvements

---

### Demonstration

- Demonstrated live system with:
  - Real-time image upload
  - Adjustable parameters
  - Immediate output generation

- Showcased:
  - Before and after images
  - Performance across different object types
  - Robustness on unseen images

---

## Final Output Capabilities

- Accurate subject extraction across multiple categories
- Clean background removal
- High-resolution output (same as input size)
- User-controlled segmentation behavior
- Interactive and responsive interface

---

## Conclusion

Milestone 4 successfully completes the VisionExtract project by delivering a fully functional application. The integration of preprocessing, model inference, and user interaction demonstrates a practical and deployable solution for subject isolation using deep learning.
#  Author

**Jafina Zeenath**
