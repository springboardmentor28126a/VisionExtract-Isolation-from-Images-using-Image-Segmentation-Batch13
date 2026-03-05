📌 Project Objective
The goal of this project is to build a machine learning model capable of automatically extracting the main subject from an image. For any given input, the model renders only the subject visible while the background is set to completely black. This subject isolation process is a core component for automation in photography, digital art,virtual meetings, and augmented reality.

Dataset Information
Primary Dataset: COCO 2017.
Subset Used: val2017.
Data Split: Shuffled and divided into 70% Train, 15% Test, and 15% Val.
Annotation Format: instances_val2017.json using pycocotools.

Project File Structure:
VISIONEXTRACT/
├── data/
│   ├── val2017/                 # Original COCO val2017 source images
│   ├── annotations/             # COCO JSON annotation files
│   └── processed/               # Standardized dataset for model training
│       ├── train/               # Training set (70%) - Includes Augmentations
│       │   ├── original images (*.jpg)
│       │   ├── *_resized.jpg    # Resized to 256x256
│       │   ├── *_normalized.jpg # Pixel range [0, 1]
│       │   ├── *_hflip.jpg      # Horizontal flip augmentation
│       │   ├── *_vflip.jpg      # Vertical flip augmentation
│       │   ├── *_bright.jpg     # Brightness adjustment augmentation
│       │   ├── *_rot30.jpg      # 30-degree rotation augmentation
│       │   ├── *_mask.png       # Binary segmentation mask
│       │   └── *_isolated.jpg   # Subject isolated with black background
│       │
│       ├── val/                 # Validation set (15%)
│       │   ├── original images (*.jpg)
│       │   ├── *_resized.jpg
│       │   ├── *_normalized.jpg
│       │   ├── *_mask.png
│       │   └── *_isolated.jpg
│       │
│       └── test/                # Test set (15%)
│           ├── original images (*.jpg)
│           ├── *_resized.jpg
│           ├── *_normalized.jpg
│           ├── *_mask.png
│           └── *_isolated.jpg
│
├── models/
│   └── vision_extract_best.keras
│   └── vision_extract_fine_tuning.keras
├── notebooks/
│   └── dataset_exploration.ipynb 
├── requirements/  
│   └── requirements.txt         # Python dependencies
├── src/
│   └── split_dataset.py          # Logic for 70/15/15 data partitioning
│   └── model.py 
│   └── utils.py 
│   └── utils1.py 
└── .gitignore                    # Files ignored by Git

✅ Milestone 1: 
Progress Summary

Week 1: 
Initialization & Dataset Inspection
Successfully loaded COCO annotations and explored image/annotation relationships.
Performed Exploratory Data Analysis (EDA) to understand object categories and mask structures.
Verified data by displaying original images alongside generated segmentation masks.

Week 2: 
Preprocessing & Binary Masking:

Binary Mask Conversion:
Raw COCO annotations contain multiple object classes. For this project, these were converted into binary masks.
Subject Pixels: Set to 1 (Foreground).
Background Pixels: Set to 0 (Black).
Subject Isolation: By applying the generated binary mask to the original image,the subject remains in its original color while all background pixels are rendered black.

Preprocessing Pipeline:.
Resizing: All inputs were resized to 256*256$ pixels.
Normalization: Pixel intensities were scaled from [0, 255] to [0, 1] to improve training stability.
Augmentation: flipping,rotation,brightening for training set.

🎯 Outcomes Achieved
Dataset successfully loaded and inspected.
Binary segmentation masks generated.
Preprocessing pipeline established (Resize, Normalize, Augment).
Subject isolation logic verified.


✅ Milestone 2: 
VisionExtract: Isolation from Images using Image Segmentation
VisionExtract is a deep learning system designed for precise subject isolation. By utilizing a custom-built U-Net architecture, the model learns to generate binary masks that separate foreground subjects from complex backgrounds, specifically optimized using the COCO 2017 dataset(val2017).


🏗 Model Architecture (model.py)
The model is a deep Convolutional Neural Network (CNN) based on the U-Net design, optimized for a 256 x 256 input resolution.
Encoder: 4 levels of double convolutions, batch normalization, and max-pooling to extract hierarchical features.
Bottleneck: The deepest layer (16, 16, 1024) captures abstract global context.
Decoder: Uses Transposed Convolutions and Skip Connections to recover spatial resolution for pixel-perfect masks.
Final Layer: 1 x 1 Convolution with Sigmoid activation for binary segmentation.


🎲 Dice Loss
Definition: Dice Loss is a loss function based on the Sørensen–Dice coefficient, which is a statistic used to gauge the similarity of two samples. In image segmentation, it measures the overlap between two samples by taking twice the area of overlap and dividing it by the sum of the total number of pixels in both masks.

🏗️ Intersection over Union (IoU)
Definition: Intersection over Union (also known as the Jaccard Index) is a metric used to measure the accuracy of an object detector or segmentation model on a particular dataset. It calculates the ratio between the area of overlap and the area of union between the predicted segmentation map and the ground truth mask.


📊 Training Strategy

1.Initial Training (utils.py)
Focused on basic convergence using normalized image-mask pairs.
Optimizer: Adam (LR = 1e-4)
Batch Size: 16
Epochs: 15
Validation Result: Accuracy ~80%, Mean IoU ~0.35, Dice Loss ~0.32.

2.Fine-Tuning with Augmentation (utils1.py)
To improve robustness, we introduced a second phase with a lower learning rate and dynamic data augmentation.
Optimizer: Adam (LR = 1e-5)
Batch Size: 16
Epochs: 5
Augmentations: Brightness shifts and Horizontal flips.
Metrics: Added Class-Specific IoU (bg_iou and subject_iou).

Final Validation Result
Accuracy 78.68%
Subject IoU 51.15%
Background IoU 72.42%
Mean IOU 61.79%
Dice Loss 0.3230


🖼 Outcomes
The model successfully isolates subjects even in images with overlapping elements. The fine-tuning phase significantly improved the model's ability to "see" smaller subjects.


🛠 Tech Stack
Core: Python, TensorFlow, Keras
Processing: OpenCV, NumPy
Visualization: Matplotlib
Dataset: COCO 2017 (val2017)