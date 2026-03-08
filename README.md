# VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13
Objective:
The goal of this project is to automatically extract the main subject from an image.
The background is removed and replaced with a black background using image segmentation.

Dataset:
COCO 2017 Dataset
- Total images used: 5000 (val2017)
- Annotation file: instances_val2017.json
Week 1 Work Done
Understood the project problem statement and objectives
Downloaded and extracted the COCO 2017 validation dataset
Explored image files and corresponding annotation structure
Studied how subject masks are represented in COCO annotations
Prepared project folder structure for images, masks, and scripts

Observation:
COCO annotations store object masks as polygon coordinates which can be converted into binary masks representing subject regions.

Week 2 Work Done
Implemented a script to convert COCO polygon annotations into binary segmentation masks
Generated masks where white represents the subject and black represents the background
Verified mask generation by visually comparing images and masks
Organized dataset into structured directories for training
Implemented preprocessing steps including image resizing and normalization

Observation:
Binary masks correctly highlighted subject regions, enabling the dataset to be used for training a segmentation model.

Week 3 Work Done
Implemented the UNet architecture for semantic segmentation using PyTorch
Built a training pipeline using custom Dataset and DataLoader
Trained the model using mini-batch gradient descent with batch size 8
Used Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss) for optimization
Monitored training and validation loss to ensure proper learning
Saved the trained model weights for further evaluation

Observation:
The model started learning object structures and produced preliminary segmentation predictions.

Week 4 Work Done
Split the dataset into 70% training, 15% validation, and 15% testing
Evaluated model performance using Dice Score and Intersection over Union (IoU) metrics
Implemented an evaluation pipeline to test the trained model
Visualized predicted segmentation masks alongside ground truth masks
Improved training using combined BCE and Dice loss for better segmentation accuracy

Observation:
The model successfully learned to isolate foreground subjects and produced segmentation masks with measurable overlap with ground truth.
Visualized predicted segmentation masks alongside ground truth masks

Improved training using combined BCE and Dice loss for better segmentation accuracy
