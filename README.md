# VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13
VisionExtract – Milestone 1
📌 Project Objective

The goal of this project is to build a subject isolation system using image segmentation.
For a given input image, the system should extract the main subject and make the background completely black.

📂 Dataset Used

Dataset: COCO 2017
Subset Used: val2017 (5000 images)
Annotation File: instances_val2017.json
Location in project:

dataset/
│
├── images/
│   └── val2017/
│
└── annotations/
    └── instances_val2017.json

✅ Milestone 1 – Week 1
1. Dataset Inspection

Loaded COCO annotations using pycocotools
Explored image IDs and annotation IDs
Understood relationship between images and objects

2. Visualized Sample Data

Displayed original image
Generated segmentation mask from annotations
Confirmed mask alignment with image

✅ Milestone 1 – Week 2
🔹 Binary Mask Conversion
Converted multi-class segmentation masks into binary format
Object pixels = 1
Background pixels = 0

🔹 Preprocessing Pipeline

The following preprocessing steps were implemented:
Image resizing to 256×256
Mask resizing using INTER_NEAREST to preserve binary values
Image normalization (pixel range 0–255 → 0–1)
Image-mask alignment verification

🔹 Subject Isolation

Applied binary mask to image
Generated output where:
Object pixels are preserved
Background pixels are set to black

🎯 Outcome of Milestone 1

✔ Dataset successfully loaded and inspected
✔ Binary segmentation masks generated
✔ Preprocessing pipeline established
✔ Subject isolation demonstrated

Milestone 1 completed successfully.