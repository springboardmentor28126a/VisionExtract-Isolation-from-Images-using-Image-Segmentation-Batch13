import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

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

        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.transpose(image, (2,0,1)).astype(np.float32)

        # Read mask
        mask = cv2.imread(mask_path, 0)
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image), torch.tensor(mask)
    
from torch.utils.data import DataLoader

image_dir = r"E:\COCO Dataset\processed_val\masked_images"
mask_dir  = r"E:\COCO Dataset\processed_val\masks"

dataset = SegmentationDataset(image_dir, mask_dir)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
import segmentation_models_pytorch as smp
import torch

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
import torch.nn as nn

def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))
def iou_score(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 30

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_iou = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = dice_loss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_iou += iou_score(outputs, masks).item()

    print(f"Epoch {epoch+1}/{epochs}")
    print("Loss:", epoch_loss/len(train_loader))
    print("IoU :", epoch_iou/len(train_loader))
# import matplotlib.pyplot as plt

# model.eval()
# images, masks = next(iter(train_loader))
# images = images.to(device)

# with torch.no_grad():
#     preds = model(images)
#     preds = torch.sigmoid(preds)
#     preds = (preds > 0.5).float()

# img = images[0].cpu().numpy().transpose(1,2,0)
# true_mask = masks[0][0].numpy()
# pred_mask = preds[0][0].cpu().numpy()

# plt.figure(figsize=(12,4))

# plt.subplot(1,3,1)
# plt.imshow(img)
# plt.title("Image")
# plt.axis("off")

# plt.subplot(1,3,2)
# plt.imshow(true_mask, cmap="gray")
# plt.title("Ground Truth")
# plt.axis("off")

# plt.subplot(1,3,3)
# plt.imshow(pred_mask, cmap="gray")
# plt.title("Prediction")
# plt.axis("off")

# plt.show()