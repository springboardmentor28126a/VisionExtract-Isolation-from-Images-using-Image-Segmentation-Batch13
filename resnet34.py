# =========================================
# WEEK 5: COCO SEGMENTATION (FULL VERSION)
# Includes: Loss + IoU + Dice
# =========================================

# ========= 1. IMPORTS =========
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from pycocotools.coco import COCO

# ========= 2. PATH CONFIG =========
COCO_ROOT = r"C:\Users\Thanuja\Desktop\vision_extraction\COCO Dataset\coco_2017"

IMAGE_DIR = os.path.join(COCO_ROOT, "train2017")
ANNOTATION_FILE = os.path.join(COCO_ROOT, "annotations", "instances_train2017.json")

assert os.path.exists(IMAGE_DIR), "Image folder not found!"
assert os.path.exists(ANNOTATION_FILE), "Annotation file not found!"

# ========= 3. LOAD COCO =========
coco = COCO(ANNOTATION_FILE)
image_ids = coco.getImgIds()

# Use 5000 images
image_ids = image_ids[:5000]

print(f"✅ Loaded {len(image_ids)} images")

# ========= 4. DATASET CLASS =========
class CocoDataset(Dataset):
    def __init__(self, coco, image_dir, image_ids, transform=None):
        self.coco = coco
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(img_path)

        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image), torch.tensor(mask)

# ========= 5. AUGMENTATION =========
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=25, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
])

# ========= 6. DATA LOADER =========
dataset = CocoDataset(coco, IMAGE_DIR, image_ids, transform=train_transform)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

# ========= 7. DEVICE =========
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ========= 8. MODEL =========
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

model.to(device)

# ========= 9. LOSS =========
bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

def loss_fn(pred, target):
    return bce(pred, target) + dice_loss(pred, target)

# ========= 10. METRICS =========
def calculate_iou(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection + 1e-6

    return (intersection + 1e-6) / union

def calculate_dice(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum()
    return (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

# ========= 11. OPTIMIZER =========
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ========= 12. TRAIN FUNCTION =========
def train_one_epoch():
    model.train()
    total_loss = 0
    total_iou = 0
    total_dice = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += calculate_iou(preds, masks).item()
        total_dice += calculate_dice(preds, masks).item()

    return (
        total_loss / len(loader),
        total_iou / len(loader),
        total_dice / len(loader)
    )

# ========= 13. TRAIN LOOP =========
EPOCHS = 15

for epoch in range(EPOCHS):
    loss, iou, dice = train_one_epoch()

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {loss:.4f} | "
          f"IoU: {iou:.4f} | "
          f"Dice: {dice:.4f}")

# ========= 14. SAVE MODEL =========
torch.save(model.state_dict(), "week5_coco_model.pth")
print("✅ Model saved successfully!")

def save_sample_results(dataset, num_samples=5):
    model.eval()

    for i in range(num_samples):
        image, mask = dataset[i]

        # Move to device
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image_tensor)
            pred = torch.sigmoid(pred).cpu().numpy()[0, 0]

        # Convert tensors to numpy
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()

        # Threshold prediction
        pred_mask = (pred > 0.5).astype(np.uint8)

        # Resize if needed
        pred_mask_resized = cv2.resize(pred_mask, (image_np.shape[1], image_np.shape[0]))

        # Convert image back to uint8
        image_np = (image_np * 255).astype(np.uint8)

        # Final output (subject only)
        final_output = image_np * np.expand_dims(pred_mask_resized, axis=2)

        # Save files
        cv2.imwrite(f"sample_{i}_input.png", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"sample_{i}_ground_truth.png", mask_np * 255)
        cv2.imwrite(f"sample_{i}_predicted_mask.png", pred_mask_resized * 255)
        cv2.imwrite(f"sample_{i}_final_output.png", cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))

    print("✅ Sample images saved successfully!")