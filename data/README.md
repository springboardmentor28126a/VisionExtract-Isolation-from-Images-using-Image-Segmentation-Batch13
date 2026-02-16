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
├── notebooks/
│   └── dataset_exploration.ipynb # EDA and mask inspection
├── requirements/  
│    └── requirements.txt         # Python dependencies
├── src/
│   └── split_dataset.py         # Logic for 70/15/15 data partitioning
└── .gitignore                   # Files ignored by Git

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