

# VisionExtract

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-green)
![Model](https://img.shields.io/badge/Model-U--Net%20%7C%20DeepLabV3+-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Subject Isolation from Images using Deep Learning Segmentation

VisionExtract is a computer vision project that performs binary semantic segmentation to isolate the main subject from an image. It uses deep learning techniques to generate a segmentation mask and remove the background, producing a clean subject-focused output.

---

## Project Overview

This project builds an end-to-end pipeline for subject extraction using a deep learning segmentation model. It covers dataset preparation, model training, evaluation, optimization, and deployment through an interactive interface.

---

## Features

* **Binary semantic segmentation** for subject isolation
* **Background removal** from images
* **High-quality mask generation**
* Supports **real-world unseen images**
* **Interactive web interface** using Gradio
* **Transparent background** output support

---

## Project Pipeline

```text
COCO Dataset + Annotations
           ↓
Polygon to Binary Mask Conversion
           ↓
    Data Preprocessing
           ↓
Train / Validation / Test Split
           ↓
Model Training (U-Net / DeepLabV3+)
           ↓
   Evaluation using IoU Metric
           ↓
      Model Optimization
           ↓
      Inference Pipeline
           ↓
   Subject Extraction Output
```

---

## Tech Stack

* **Language:** Python
* **Framework:** PyTorch, segmentation-models-pytorch
* **Libraries:** NumPy, OpenCV, Matplotlib
* **Dataset:** COCO Dataset
* **Deployment:** Gradio

---

## Dataset

The project uses a subset of the **COCO val2017** dataset.

### Structure
```text
val2017/
   ├── image1.jpg
   ├── image2.jpg
   └── ...

annotations/
   └── instances_val2017.json
```

### Processing
* Polygon annotations converted to binary masks
* **Person class** mapped to 1
* **Background** mapped to 0

---

## Data Preparation

* Filtered person category.
* Converted annotations into pixel-level masks.
* Applied resizing and normalization.
* Verified image-mask alignment.
* **Dataset Split:**
    * Train (70%)
    * Validation (15%)
    * Test (15%)
* Built custom `Dataset` and `DataLoader` classes.

---

## Model Architecture

The project primarily uses **U-Net** for segmentation.

### Key Characteristics
* Encoder-decoder architecture.
* **Skip connections** for better spatial learning.
* Pixel-wise prediction.

### Encoder
* **ResNet34** pretrained on ImageNet.

---

## Training Details

* **Loss Function:** Binary Cross Entropy (BCE) with Logits Loss
* **Optimizer:** Adam Optimizer
* **Data Augmentation:**
    * Horizontal and vertical flips
    * Rotation
    * Color jitter

---

## Evaluation Metric

**Intersection over Union (IoU)**

$$IoU = \frac{\text{Intersection}}{\text{Union}}$$

> Higher IoU indicates better segmentation performance.

---

## Model Optimization

* **Hyperparameter tuning:** learning rate, batch size, epochs.
* **Architecture comparison:**
    * U-Net + ResNet34
    * U-Net + ResNet50
    * DeepLabV3+

Evaluation based on accuracy, inference speed, and mask quality.

---

## Post-Processing

Morphological operations are applied to refine the output:
* **Closing:** To fill small gaps within the subject.
* **Opening:** To remove background noise.

This improves mask clarity and boundary quality.

---

## Inference Pipeline

1.  Load trained model
2.  Preprocess input image
3.  Generate segmentation mask
4.  Apply thresholding
5.  Refine mask (Post-processing)
6.  Resize to original resolution
7.  Extract subject
8.  Save output image

---

## User Interface

Built using **Gradio**.

### Capabilities
* Upload image
* Adjust segmentation threshold
* Control edge smoothing
* Enable transparent background
* View results instantly

---

## Output

* Subject-isolated images.
* Clean background removal.
* High-resolution output.
* Works on diverse real-world inputs.

---

## Installation

```bash
git clone [https://github.com/your-username/VisionExtract.git](https://github.com/your-username/VisionExtract.git)
cd VisionExtract
pip install -r requirements.txt
```

---

## Usage

```bash
python app.py
```
Then open the local Gradio link provided in your terminal.

---

## Project Structure

```text
VisionExtract/
├── data/
├── models/
├── notebooks/
├── utils/
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Future Work

* Multi-class segmentation
* Real-time video segmentation
* Cloud web deployment (Hugging Face Spaces)
* Mobile optimization (TorchScript/ONNX)
* Higher resolution training

---

## Author

**Jafina Zeenath**
