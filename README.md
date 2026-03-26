# VisionExtract – Subject Isolation from Images using Image Segmentation

## Objective
The goal of this project is to automatically extract the main subject from an image using deep learning-based image segmentation. The background is removed and replaced with a black background.

---

## Dataset
- Dataset Used: COCO 2017 Dataset  
- Subset: Validation Set (val2017)  
- Total Images Used: 5000  
- Annotation File: instances_val2017.json  

---

## Week-wise Progress

### Week 1 – Project Setup & Understanding

**Work Done**
- Understood project objectives and problem statement  
- Downloaded and extracted COCO 2017 dataset  
- Explored dataset structure (images + annotations)  
- Studied COCO polygon annotation format  
- Created project folder structure  

**Observation**  
COCO annotations store object masks as polygon coordinates, which can be converted into binary segmentation masks.

---

### Week 2 – Data Preprocessing & Mask Generation

**Work Done**
- Converted COCO polygon annotations into binary masks  
- Generated masks (white = subject, black = background)  
- Verified masks visually with original images  
- Organized dataset into structured directories  
- Implemented preprocessing (resize + normalization)  

**Observation**  
Binary masks correctly represent subject regions and are suitable for training segmentation models.

---

### Week 3 – Model Implementation & Training

**Work Done**
- Implemented U-Net architecture using PyTorch  
- Built custom Dataset and DataLoader  
- Trained model using batch size = 8  
- Used BCEWithLogitsLoss  
- Monitored training and validation loss  
- Saved trained model weights  

**Observation**  
Model started learning object structures and produced initial segmentation predictions.

---

### Week 4 – Evaluation & Performance Improvement

**Work Done**
- Split dataset (70% train / 15% val / 15% test)  
- Implemented evaluation pipeline  
- Used Dice Score and IoU metrics  
- Visualized predictions vs ground truth  
- Improved training using BCE + Dice Loss  

**Observation**  
Model successfully learned foreground separation and produced measurable segmentation accuracy.
