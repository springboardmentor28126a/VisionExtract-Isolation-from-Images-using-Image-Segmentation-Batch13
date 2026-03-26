**VisionExtract – Subject Isolation from Images using Image Segmentation
Objective**

The goal of this project is to automatically extract the main subject from an image using deep learning-based image segmentation. The background is removed and replaced with a black background.

Dataset
Dataset Used: COCO 2017 Dataset
Subset: Validation Set (val2017)
Total Images Used: 5000
Annotation File: instances_val2017.json
Week-wise Progress
Week 1 – Project Setup & Understanding
Work Done
Understood project objectives and problem statement
Downloaded and extracted COCO 2017 dataset
Explored dataset structure (images + annotations)
Studied COCO polygon annotation format
Created project folder structure
Observation

COCO annotations store object masks as polygon coordinates, which can be converted into binary segmentation masks.

Week 2 – Data Preprocessing & Mask Generation
Work Done
Converted COCO polygon annotations into binary masks
Generated masks (white = subject, black = background)
Verified masks visually with original images
Organized dataset into structured directories
Implemented preprocessing (resize + normalization)
Observation

Binary masks correctly represent subject regions and are suitable for training segmentation models.

Week 3 – Model Implementation & Training
Work Done
Implemented U-Net architecture using PyTorch
Built custom Dataset and DataLoader
Trained model using batch size = 8
Used BCEWithLogitsLoss
Monitored training and validation loss
Saved trained model weights
Observation

Model started learning object structures and produced initial segmentation predictions.

Week 4 – Evaluation & Performance Improvement
Work Done
Split dataset (70% train / 15% val / 15% test)
Implemented evaluation pipeline
Used Dice Score and IoU metrics
Visualized predictions vs ground truth
Improved training using BCE + Dice Loss
Observation

Model successfully learned foreground separation and produced measurable segmentation accuracy.

Week 5 – Model Improvement & Optimization
Work Done
Upgraded model from U-Net to DeepLabV3 (ResNet-50 backbone)
Added advanced data augmentation:
Horizontal & vertical flips
Rotation
Brightness & contrast adjustments
Switched to AdamW optimizer for better generalization
Added learning rate scheduler
Used mixed precision training (AMP) for faster GPU training
Improved mask quality using post-processing techniques
Observation

The improved model achieved better segmentation quality with cleaner edges and higher Dice/IoU scores compared to the baseline.

Week 6 – Inference & Deployment
Work Done
Developed an inference pipeline for unseen images
Applied post-processing techniques:
Thresholding
Morphological operations
Largest connected component filtering
Implemented subject isolation using predicted masks
Supported single image and batch processing
Built a Gradio web interface for user interaction
Observation

The system successfully generates subject-isolated images from new inputs and provides an interactive interface for real-time testing.

**Model Architecture**
Model Used: DeepLabV3
Backbone: ResNet-50
Output: Single-channel binary segmentation mask
Evaluation Metrics
Dice Score
Intersection over Union (IoU)
**Technologies Used**
Python
PyTorch
OpenCV
NumPy
Matplotlib
Gradio

**Final Pipeline**
Input Image
     ↓
Preprocessing
     ↓
DeepLabV3 Model
     ↓
Segmentation Mask
     ↓
Post-processing
     ↓
Subject Isolation
     ↓
Output Image
Result

The final system successfully removes the background and isolates the main subject from images with high accuracy and provides an interactive interface for testing.

**Future Work**
Improve edge refinement using advanced matting techniques
Deploy model as a web application
Optimize inference speed for real-time applications
