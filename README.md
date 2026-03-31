# VisionExtract – Subject Isolation using Image Segmentation

## Project Description

VisionExtract is a deep learning project designed to automatically isolate the main subject from an image using image segmentation techniques.

Given an input image, the system predicts a binary segmentation mask that identifies the subject region. The output image retains the subject while converting the background pixels to black.

This project demonstrates the complete machine learning pipeline for an image segmentation system including:

- Dataset acquisition and exploration  
- Data preprocessing and preparation  
- Deep learning model development  
- Training and prediction generation  
- Model improvement and optimization  
- Deployment using a web application  

---

## Dataset Used: COCO 2017 Dataset

---

# 🔹 Milestone 1 – Data Preprocessing

## Objective

The goal of this milestone is to prepare the dataset for training the segmentation model. Raw images and masks must be processed to ensure consistency and correct alignment.

## Key Tasks

- Inspect dataset structure and annotations  
- Preprocess images and segmentation masks  
- Resize images to a consistent dimension  
- Normalize image pixel values  
- Convert masks into binary subject/background format  
- Shuffle dataset samples for better model generalization  

## Implementation Files

### preprocessing.py
Responsible for:
- Image resizing  
- Image normalization  
- Mask formatting  
- Preparing input tensors  

### shuffling.py
Responsible for:
- Randomizing dataset samples  
- Preventing training bias  

## Expected Output

- Clean image–mask pairs  
- Consistent resolution  
- Binary masks  
- Training-ready dataset  

---

# 🔹 Milestone 2 – Model Training

## Objective

To build and train a deep learning segmentation model capable of extracting the subject region.

## Model Used

- U-Net Architecture (initial model)

## Key Tasks

- Implement U-Net architecture  
- Train model on dataset  
- Generate segmentation masks  
- Evaluate on validation data  

## Implementation Files

### unet_model.py
Defines:
- Encoder  
- Decoder  
- Skip connections  

### unet_implementation.py
Handles:
- Training loop  
- Loss calculation  
- Optimization  
- Prediction generation  

## Evaluation Metrics

- Intersection over Union (IoU)  
- Dice Coefficient  
- Pixel Accuracy  

---

# 🔹 Milestone 3 – Model Improvement & Inference

## Objective

To improve segmentation accuracy and perform robust inference on unseen images.

## Improvements Made

- Switched to **DeepLabV3 with ResNet50 backbone**
- Better feature extraction using pretrained weights  
- Combined **BCE Loss + Dice Loss**  
- Improved generalization on complex backgrounds  

## Key Tasks

- Load pretrained DeepLabV3 model  
- Fine-tune on dataset  
- Implement validation pipeline  
- Generate improved segmentation outputs  

## Inference Enhancements

- Resize prediction to original image size  
- Apply sigmoid activation  
- Thresholding for mask generation  
- Gaussian smoothing for noise reduction  
- Largest connected component extraction  
- Morphological operations (open & close)  

## Output

- High-quality segmentation masks  
- Cleaner subject isolation  
- Reduced background noise  

---

# 🔹 Milestone 4 – Model Deployment (Streamlit Web App)

## Objective

To deploy the trained segmentation model into an interactive web application for real-time usage.

## Features

- Upload image through UI  
- Real-time background removal  
- Subject isolation using trained model  
- Custom background options (black/white/custom)  
- Download processed output  
- Clean and user-friendly interface  

## Technologies Used

- Streamlit (Frontend + Backend)  
- PyTorch (Model Inference)  
- OpenCV (Image Processing)  

## Implementation

### app.py

Handles:
- User interface  
- Image upload  
- Model loading  
- Inference pipeline  
- Displaying results  
- Download functionality  

## Output

- Original image preview  
- Segmented output image  
- Downloadable result  

---

```
# 📁 Project Directory Structure
VisionExtract_Segmentation
│
├── 01_data_preprocessing
│ ├── preprocessing.py
│ └── shuffling.py
│
├── 02_model_training
│ ├── unet_model.py
│ └── unet_implementation.py
│
├── 03_model_improvement_and_inference
│ ├── deeplab_model.py
│ ├── training.py
│ └── inference.py
│
├── 04_model_streamlit_app
│ └── app.py
│
├── 05_results
│ ├── preprocessing_results
│ ├── segmentation_output.png
│ └── training_output.png
│ ├── model_training_result.png
│ ├── inference_result.png
│ └── app_interface.png
│
└── README.md
```

---

# 🎯 Final Outcome

- Built an end-to-end image segmentation pipeline  
- Improved model performance using DeepLabV3  
- Successfully deployed model as a web application  
- Enabled real-time subject isolation  

---

# 🧠 Conclusion

VisionExtract demonstrates a complete deep learning workflow from data preprocessing to deployment. The integration of advanced segmentation models with an interactive interface makes it practical for real-world applications such as background removal, image editing, and object extraction.



