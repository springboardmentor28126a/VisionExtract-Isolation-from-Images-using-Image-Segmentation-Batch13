# =========================
# IMPORTS
# =========================
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# =========================
# PATHS
# =========================
BASE_PATH = "C:/Users/varsh/Downloads/archive/coco2017/split_data/processed"

TRAIN_IMG = os.path.join(BASE_PATH, "train/images")
TRAIN_MASK = os.path.join(BASE_PATH, "train/masks")

VAL_IMG = os.path.join(BASE_PATH, "val/images")
VAL_MASK = os.path.join(BASE_PATH, "val/masks")

# =========================
# DATASET
# =========================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # IMAGE
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256)) / 255.0

        # MASK
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 127).astype(np.float32)

        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask

# =========================
# LOAD DATA
# =========================
train_loader = DataLoader(
    SegmentationDataset(TRAIN_IMG, TRAIN_MASK),
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    SegmentationDataset(VAL_IMG, VAL_MASK),
    batch_size=4
)

# =========================
# MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

model = model.to(device)

# =========================
# LOSS FUNCTION
# =========================
bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1

    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def loss_fn(pred, target):
    return bce(pred, target) + dice_loss(pred, target)

# =========================
# OPTIMIZER
# =========================
optimizer = optim.Adam(model.parameters(), lr=3e-5)

# =========================
# TRAINING
# =========================
num_epochs = 30

for epoch in range(num_epochs):

    # 🔵 TRAINING
    model.train()
    train_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)['out']

        outputs = F.interpolate(
            outputs,
            size=masks.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # 🟢 VALIDATION
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)['out']

            outputs = F.interpolate(
                outputs,
                size=masks.shape[2:],
                mode='bilinear',
                align_corners=False
            )

            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # 📊 PRINT RESULTS
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "deeplab_resnet50.pth")

print("✅ Training Completed and Model Saved!")
