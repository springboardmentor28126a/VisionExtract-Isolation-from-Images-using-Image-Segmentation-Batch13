# VisionExtract: Subject Isolation from Images using Image Segmentation

### Hybrid Deep Learning Approach using Mask R-CNN + DeepLabV3

---

## Project Overview

This project presents an **AI-powered Subject Isolation System** that automatically detects and extracts the main subject from an image. The system uses a **hybrid deep learning pipeline** combining **instance segmentation (Mask R-CNN)** and **semantic segmentation (DeepLabV3)** to achieve accurate and refined subject isolation.

The final system includes:

* Intelligent subject detection
* Pixel-level segmentation refinement
* Clean background removal
* Web-based user interface for real-time usage

---

## Objectives

* Develop an automated system for **subject isolation from images**
* Improve segmentation quality using a **hybrid model**
* Compare different approaches and evaluate performance
* Build a **complete pipeline from dataset to deployment**
* Provide a **user-friendly web interface**

---

## Methodology

### 🔹 Hybrid Model Approach

The project uses a **two-stage hybrid architecture**:

1. **Mask R-CNN**

   * Detects objects in the image
   * Generates instance-level masks
   * Provides object labels and confidence scores

2. **DeepLabV3**

   * Performs semantic segmentation
   * Refines boundaries at pixel level
   * Improves mask accuracy

### 🔹 Pipeline Flow

```
Input Image
     ↓
Mask R-CNN (Object Detection + Mask)
     ↓
Select Main Subject (Highest Confidence)
     ↓
DeepLabV3 (Refinement)
     ↓
Mask Combination
     ↓
Morphological Cleaning
     ↓
Final Subject Isolation
     ↓
Output Image
```

---

## Dataset

* Dataset Used: **COCO Dataset (Common Objects in Context)**
* Contains:

  * Images
  * Object annotations
  * Segmentation masks

### Dataset Processing:

* Converted multi-class masks → binary masks
* Resized images
* Normalized pixel values
* Validated image-mask alignment

---

## Technologies Used

| Category         | Tools                 |
| ---------------- | --------------------- |
| Language         | Python                |
| Deep Learning    | PyTorch               |
| Models           | Mask R-CNN, DeepLabV3 |
| Image Processing | OpenCV                |
| Visualization    | Matplotlib            |
| UI               | Gradio                |
| Platform         | Google Colab          |

---

## Implementation (Week-wise)

---

### Week 1: Project Initialization

* Defined project scope and objectives
* Selected COCO dataset
* Explored dataset structure
* Visualized images and masks

---

### Week 2: Data Preprocessing

* Image resizing and normalization
* Data augmentation:

  * Flipping
  * Rotation
  * Scaling
* Mask validation
* Binary mask conversion

---

### Week 3: Initial Model Training

* Implemented Mask R-CNN
* Trained on dataset subset
* Generated initial predictions
* Observed segmentation quality

---

### Week 4: Prediction & Fine-tuning

* Visualized predictions
* Adjusted thresholds
* Improved mask extraction
* Added preprocessing enhancements

---

### Week 5: Model Improvement

* Introduced DeepLabV3
* Built hybrid pipeline
* Compared models:

| Model        | IoU      | Dice     | Accuracy |
| ------------ | -------- | -------- | -------- |
| Mask R-CNN   | Moderate | Moderate | Good     |
| Hybrid Model | High     | High     | Better   |

---

### Week 6: Inference Pipeline

* Built automated pipeline
* Tested on unseen images
* Saved output images
* Improved robustness

---

### Week 7: Web Application

* Built Gradio UI
* Features:

  * Upload image
  * View mask & output
  * Download results
  * Show confidence & label

---

## Evaluation Metrics

### 1. IoU (Intersection over Union)

Measures overlap between predicted and actual mask.

### 2. Dice Score

Measures similarity between masks.

### 3. Pixel Accuracy

Percentage of correctly classified pixels.

---

## Results

* Hybrid model outperformed Mask R-CNN
* Better edge refinement
* Improved segmentation accuracy
* Robust on multiple object types

---

## Observations

* Mask R-CNN works well for known objects (person, car)
* DeepLabV3 improves boundary precision
* Hybrid approach gives best results
* Complex backgrounds reduce accuracy
* Performance depends on object clarity

---

## Challenges Faced

1. **Poor mask quality initially**
2. **Multiple object confusion**
3. **Black mask issue**
4. **Model compatibility issues**
5. **Colab limitations**
6. **Handling no detection cases**
7. **Combining two different model outputs**

---

## Solutions Implemented

* Used highest confidence object selection
* Added DeepLabV3 refinement
* Applied morphological operations
* Handled None masks safely
* Resized masks before combining
* Used fallback mechanisms

---

## Web Application Features

* Upload image
* Process using AI pipeline
* Display:

  * Original Image
  * Mask
  * Isolated Subject
* Show:

  * Object Label
  * Confidence Score
* Download:

  * Original Image
  * Output Image

---

## Installation & Setup

### Step 1: Clone Repository

```
git clone <your-repo-link>
cd project-folder
```

### Step 2: Install Dependencies

```
pip install torch torchvision opencv-python gradio matplotlib
```

### Step 3: Run Application

```
python app.py
```

OR in Colab:

```
app.launch()
```

---

## ▶Usage

1. Upload an image
2. Click "Process Image"
3. View output
4. Download results

---

## Sample Output

* Clean subject isolation
* Background removed
* Accurate segmentation

---

## Limitations

* Not perfect for:

  * Buildings
  * Complex scenes
  * Low-quality images
* Depends on COCO-trained classes
* Cannot detect unknown objects well

---

## Future Improvements

* Use advanced models (U²-Net)
* Real-time video processing
* Multi-object selection
* Background blur effect
* Mobile app deployment
* Custom dataset training

---

## Conclusion

This project successfully demonstrates an **AI-based subject isolation system** using a hybrid deep learning approach. The integration of Mask R-CNN and DeepLabV3 significantly improves segmentation quality and produces reliable results.

The system is practical, scalable, and showcases real-world AI application development from **data processing to deployment**.

---

## Acknowledgment

* COCO Dataset
* PyTorch Community
* Open-source contributors

---

## Author

**Himanshu Ahirrao**

---

## Final Note

This project is a complete pipeline including:

* Data preprocessing
* Model implementation
* Hybrid architecture
* Evaluation
* Deployment

It demonstrates strong understanding of:

* Computer Vision
* Deep Learning
* Software Integration

---