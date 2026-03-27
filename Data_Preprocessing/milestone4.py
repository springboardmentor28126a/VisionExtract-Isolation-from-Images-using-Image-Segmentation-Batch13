import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ================================
# Paths
# ================================

IMAGE_DIR = r"E:\COCO Dataset\processed_val_masked_after_modification\masked_images"
MASK_DIR  = r"E:\COCO Dataset\processed_val_masked_after_modification\masks"

# ================================
# Augmentation
# ================================

# train_transform = A.Compose([
#     A.Resize(256,256),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.3),
# ])
train_transform = A.Compose([
    A.Resize(512,512),

    A.HorizontalFlip(p=0.5),
    ToTensorV2()
    # A.VerticalFlip(p=0.5),

    # A.Rotate(limit=30,p=0.5),

    # A.RandomBrightnessContrast(p=0.4),
    # A.ColorJitter(p=0.3),

    # A.GaussianBlur(p=0.2),

    # A.ShiftScaleRotate(
    #     shift_limit=0.05,
    #     scale_limit=0.1,
    #     rotate_limit=20,
    #     p=0.5
    # ),
])

val_transform = A.Compose([
    A.Resize(512,512),
])

# ================================
# Dataset
# ================================

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.files = file_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img_name = self.files[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)
        # print("Mask unique values:", np.unique(mask))
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # image = image / 255.0
        # image = np.transpose(image, (2,0,1)).astype(np.float32)
        # image=image/255.0
        # mean = np.array([0.485, 0.456, 0.406])
        # std  = np.array([0.229, 0.224, 0.225])

        # image = (image - mean) / std
        # image = np.transpose(image, (2,0,1)).astype(np.float32)

        image=image.float()/255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        image = (image - mean) / std
        mask = (mask >= 128).float()
        mask = np.expand_dims(mask, axis=0)
        # print("Image shape:", image.shape)
        # print("Mask shape :", mask.shape)
        # return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
        return image, mask
# ================================
# Train / Validation Split
# ================================

files = os.listdir(IMAGE_DIR)

train_files, val_files = train_test_split(
    files,
    test_size=0.2,
    random_state=42
)

train_dataset = SegmentationDataset(
    IMAGE_DIR, MASK_DIR, train_files, train_transform
)

val_dataset = SegmentationDataset(
    IMAGE_DIR, MASK_DIR, val_files, val_transform
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ================================
# Model
# ================================

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ================================
# Loss Functions
# ================================

def dice_loss(pred, target, smooth=1):

    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

bce_loss = nn.BCEWithLogitsLoss()

# ================================
# IoU Metric
# ================================

def iou_score(pred, target, threshold=0.5):

    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection + 1e-6) / (union + 1e-6)

# ================================
# Optimizer
# ================================

optimizer = optim.Adam(model.parameters(), lr=3e-4)

# ================================
# Training Loop
# ================================

EPOCHS = 25
best_iou = 0

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0
    train_iou = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        # loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
        loss = 0.5 * bce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iou += iou_score(outputs, masks).item()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("Train Loss:", train_loss / len(train_loader))
    print("Train IoU :", train_iou / len(train_loader))

    # ==========================
    # Validation
    # ==========================

    model.eval()

    val_iou = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            val_iou += iou_score(outputs, masks).item()
    val_iou = val_iou/len(val_loader)
    print("Validation IoU:", val_iou )
    
    # Save the model if it has the best IoU so far
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), "efficientnet-b4_model.pth")
        print("Model saved successfully!")

# ================================
# Predictions on Validation Set
# ================================

model.eval()

images, masks = next(iter(val_loader))

with torch.no_grad():

    images = images.to(device)

    outputs = model(images)

    preds = torch.sigmoid(outputs)
    preds = (preds > 0.2).float()

# ================================
# Visualization
# ================================

img = images[0].cpu().numpy().transpose(1,2,0)
gt = masks[0][0].cpu().numpy()
pred = preds[0][0].cpu().numpy()
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

img = (img * std) + mean
img = np.clip(img,0,1)

plt.figure(figsize=(16,4))

plt.subplot(1,4,1)
plt.title("Original Image")
plt.imshow(img)

plt.subplot(1,4,2)
plt.title("Ground Truth Mask")
plt.imshow(gt, cmap="gray")

plt.subplot(1,4,3)
plt.title("Predicted Mask")
plt.imshow(pred, cmap="gray")

plt.subplot(1,4,4)
plt.title("Comparison")
plt.imshow(img)
# plt.imshow(1,4,5)
# plt.title("Raw Preds")
# plt.imshow(preds[0][0].cpu(), cmap="jet")
plt.show()