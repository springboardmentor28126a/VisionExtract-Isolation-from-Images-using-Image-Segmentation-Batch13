# 📘 VisionExtract: Subject Isolation from Images using Image Segmentation

---

## 📌 Project Overview

VisionExtract is a deep learning-based project that focuses on **automatically extracting the main subject from an image**.

The system takes an input image and produces an output where:

* The **main subject is preserved**
* The **background is completely blacked out**

This technique is widely useful in:

* Photography automation
* Digital art
* Augmented Reality
* Virtual conferencing
* Background replacement systems

---

## 🎯 Objective

To build a **semantic segmentation model** that performs **pixel-wise classification** to isolate the subject from the background.

---

## 🛠️ Tech Stack

* Python
* Deep Learning (TensorFlow / PyTorch)
* OpenCV
* NumPy, Pandas
* Matplotlib
* Flask / Streamlit (for UI)

---

## 📂 Dataset

* COCO 2017 Dataset
* Link: https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset

---

## 🚀 Project Milestones

---

### ✅ Milestone 1: Data Preparation

#### 📅 Week 1: Project Initialization & Dataset Acquisition

**Tasks:**

* Defined project objectives
* Downloaded and explored COCO dataset
* Visualized sample images and masks

**Output:**

* Sample dataset images
* Understanding of data structure

---

#### 📅 Week 2: Data Preprocessing & Validation

**Tasks:**

* Resized images and masks
* Normalized input data
* Applied augmentation (flip, crop, color changes)
* Converted masks into binary (subject vs background)
* Ensured image-mask alignment

**Output:**

* Cleaned and preprocessed dataset
* Binary masks ready for training

---

### ✅ Milestone 2: Model Development

#### 📅 Week 3: Initial Model Training

**Tasks:**

* Built segmentation model (e.g., U-Net / CNN-based)
* Trained model on dataset
* Monitored training metrics

**Output:**

* Initial trained model
* Early prediction results

---

#### 📅 Week 4: Prediction & Fine-tuning

**Tasks:**

* Generated predictions on validation set
* Compared outputs with ground truth masks
* Tuned hyperparameters
* Improved data augmentation

**Output:**

* Improved segmentation results
* Better subject extraction quality

---

### ✅ Milestone 3: Optimization & Inference

#### 📅 Week 5: Model Improvement

**Tasks:**

* Enhanced preprocessing techniques
* Tested different architectures
* Compared model performance

**Output:**

* Optimized model
* Performance comparison report

---

#### 📅 Week 6: Inference Pipeline

**Tasks:**

* Built inference pipeline
* Input: image → Output: subject-isolated image
* Tested on unseen images

**Output:**

* Working subject isolation system
* Output images with black background

---

### ✅ Milestone 4: Deployment & Presentation

#### 📅 Week 7: Full Pipeline & UI Development

**Tasks:**

* Developed web interface (Flask/Streamlit)
* Integrated preprocessing + model + output
* Enabled image upload functionality

**Output:**

* Functional web app
* User-upload image processing

---

#### 📅 Week 8: Documentation & Demo

**Tasks:**

* Prepared complete documentation
* Created presentation slides
* Added before/after results
* Conducted demo

**Output:**

* Final report
* PPT / Demo
* Project submission

---

## 📊 Evaluation Metrics

### 🔹 Primary Metric

* **Intersection over Union (IoU)**

  * Measures overlap between predicted mask and ground truth

### 🔹 Additional Metrics

* Dice Coefficient
* Pixel-wise Accuracy
* Precision & Recall

---

## ⚙️ Workflow

1. Input Image
2. Preprocessing
3. Segmentation Model Prediction
4. Binary Mask Generation
5. Apply Mask → Extract Subject
6. Output Image (Black Background)

---

## ▶️ How to Run

```bash
# Clone repository
git clone https://github.com/your-username/visionextract.git

# Navigate to project
cd visionextract

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

---

## 📸 Results

* Input Image → Original photo
* Output Image → Subject isolated with black background
