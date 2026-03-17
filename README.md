# VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13

## 📌 Milestone 1 – Project Initialization & Data Preparation

### 🔍 Project Objective
The goal of this project is to build a deep learning model capable of automatically isolating selected subjects (person, animals, fruits) from an image using semantic segmentation. The output image retains only the detected subject while rendering the background completely black.

---

## ✅ Milestone 1 Completed Tasks

### 1️⃣ Dataset Acquisition
- Downloaded **COCO 2017 Dataset**
- Loaded annotations using `pycocotools`
- Explored dataset structure (images + segmentation masks)

### 2️⃣ Category Selection
Selected the following object categories for subject isolation:
- **Person**
- **Animals** (dog, cat, horse, sheep, cow, elephant, etc.)
- **Fruits / Food Items** (banana, apple, orange, pizza, etc.)

### 3️⃣ Mask Generation
- Extracted segmentation annotations from COCO
- Converted multi-class masks into **binary masks**
    - `1` → Selected object
    - `0` → Background
- Ensured correct image–mask alignment

### 4️⃣ Data Preprocessing
- Resized images and masks to **256×256**
- Normalized image pixel values (0–1 range)
- Converted masks to single-channel binary format
- Implemented custom PyTorch Dataset class

### 5️⃣ DataLoader Setup
- Created training and validation splits
- Implemented PyTorch DataLoader for batch processing

---

## 🛠 Tools & Technologies Used
- Python 3.x
- Google Colab (GPU enabled)
- PyTorch
- Segmentation Models PyTorch
- COCO Dataset
- pycocotools
- OpenCV
- NumPy
- Matplotlib

---

## 📊 Sample Binary Mask Output
<img width="1320" height="1190" alt="image" src="https://github.com/user-attachments/assets/17b1912e-d1a5-4233-a332-fdad36721008" />
---

# 📌 Milestone 2 – Model Training, Evaluation & Fine-Tuning

## 🎯 Objective
The objective of Milestone 2 was to implement, train, evaluate, and optimize a semantic segmentation model capable of accurately isolating selected subjects (person, animals, fruits) from images.

---

## 🧠 Model Architecture

We implemented **U-Net with a ResNet34 encoder** using `segmentation_models_pytorch`.

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)
```

- Encoder: Pretrained ResNet34 (ImageNet)
- Output: Single-channel binary segmentation mask

---

## ⚙️ Training Configuration

- Loss Function: `BCEWithLogitsLoss`
- Optimizer: `Adam`
- Learning Rate: `1e-4`
- Batch Size: `4`
- Image Resolution: `256×256`
- Epochs: 10

```python
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

---

## 🔁 Model Training

The model was trained using forward propagation, loss computation, backpropagation, and optimizer updates.

```python
for epoch in range(5):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
```

Training loss decreased progressively across epochs, indicating effective learning.

---

## 📊 Model Evaluation – IoU Metric

Performance was evaluated using **Intersection over Union (IoU)** on the validation set.

```python
def iou_score(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()
```

Validation Process:

```python
model.eval()
total_iou = 0

with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        total_iou += iou_score(outputs, masks)

print("Validation IoU:", total_iou/len(val_loader))
```

---

## 🖼 Output Generation

The model generates:
<img width="1320" height="1190" alt="image" src="https://github.com/user-attachments/assets/188f38b2-861b-4249-afb3-ebe1d5de7d35" />
---


## 📈 Results

- Training loss decreased across epochs.
- The model successfully detected:
  - Person
  - Animals
  - Fruits/Food items
- Background was removed using predicted segmentation masks.
- Validation IoU achieved: **(Add your final score here)**

---

## 🚀 Future Improvements

- Combine Dice Loss with BCE Loss
- Add data augmentation
- Train for more epochs
- Improve boundary sharpness
- Deploy as a web application

---

# 📌 Milestone 3 – Model Improvement & Inference Pipeline

## 🎯 Objective
The goal of Milestone 3 is to improve the trained segmentation model from Milestone 2 and build an inference pipeline that can isolate subjects from new, unseen images.

---

## 🔄 Improvements Over Milestone 2

- Continued training from pre-trained model (`final_model.pth`)
- Added **data augmentation** to improve generalization
- Introduced **Dice + BCE combined loss function**
- Improved segmentation accuracy and mask quality
- Built a complete **inference pipeline**

---

## 🧠 Model Enhancement

### 🔹 Data Augmentation
To improve robustness, the following augmentations were applied:
- Horizontal Flip
- Random Brightness & Contrast
- Rotation & Scaling

```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(p=0.3)
])
```

### 🔹 Improved Loss Function (Dice + BCE)

```python
def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-6

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice

bce = nn.BCEWithLogitsLoss()

def combined_loss(pred, target):
    return bce(pred, target) + dice_loss(pred, target)
```

### 🔹 Continued Training

```python
for epoch in range(5):
    model.train()

    for images, masks in train_loader:
        outputs = model(images)
        loss = combined_loss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 🚀 Inference Pipeline

The model can now process **new images** outside the dataset.

### 🔹 Steps

1. Upload image  
2. Preprocess image (resize + normalize)  
3. Predict segmentation mask  
4. Apply mask to isolate subject  

---

## 🖼 Output

The system generates:
<img width="1182" height="384" alt="image" src="https://github.com/user-attachments/assets/79efdafb-7619-4c6c-9846-396df1d3d010" />

---

## 📊 Results

- Improved segmentation performance compared to Milestone 2  
- Better boundary detection due to Dice Loss  
- Model successfully generalizes to **new unseen images**  
- Background effectively removed using predicted masks  

---

## 👨‍💻 Author
Rahul Raj  

---
