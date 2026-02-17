# VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13
Objective:
The goal of this project is to automatically extract the main subject from an image.
The background is removed and replaced with a black background using image segmentation.

Dataset:
COCO 2017 Dataset
- Total images used: 5000 (val2017)
- Annotation file: instances_val2017.json

Week 1 Work Done:
- Understood the project problem statement and objectives
- Downloaded and extracted the COCO 2017 validation dataset
- Explored image files and corresponding annotation structure
- Studied how subject masks are represented in COCO annotations
- Prepared project folder structure

Observation:
COCO annotations store object masks as polygon coordinates
which can be used to generate subject masks.
