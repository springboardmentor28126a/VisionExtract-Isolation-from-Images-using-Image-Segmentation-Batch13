import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
import torch.optim as optim


# ==============================
# 1. Dataset
# ==============================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))  # IMPORTANT

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # IMAGE
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        image = np.transpose(image, (2, 0, 1))

        # MASK
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()


# ==============================
# MAIN FUNCTION (VERY IMPORTANT)
# ==============================
def main():

    # Paths
    image_dir = r"E:\COCO Dataset\processed_binary\images"
    mask_dir  = r"E:\COCO Dataset\processed_binary\masks"

    dataset = SegmentationDataset(image_dir, mask_dir)

    # Split
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size   = int(0.15 * total_size)
    test_size  = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    print(f"Train: {len(train_dataset)}")
    print(f"Val  : {len(val_dataset)}")
    print(f"Test : {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)

    # Loss
    bce_loss = torch.nn.BCEWithLogitsLoss()

    def combined_loss(pred, target):
        bce = bce_loss(pred, target)
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = 1 - (2. * intersection + 1) / (pred.sum() + target.sum() + 1)
        return bce + dice

    # Metrics
    def iou_score(pred, target):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum(dim=(1,2,3))
        union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection
        return ((intersection + 1e-6) / (union + 1e-6)).mean()

    def pixel_accuracy(pred, target):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        return (pred == target).float().mean()

    # Training
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 30

    best_iou = 0

    for epoch in range(epochs):

        model.train()
        train_loss, train_iou, train_acc = 0, 0, 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = combined_loss(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou  += iou_score(outputs, masks).item()
            train_acc  += pixel_accuracy(outputs, masks).item()

        model.eval()
        val_iou, val_acc = 0, 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                val_iou += iou_score(outputs, masks).item()
                val_acc += pixel_accuracy(outputs, masks).item()

        train_loss /= len(train_loader)
        train_iou  /= len(train_loader)
        train_acc  /= len(train_loader)

        val_iou /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train IoU : {train_iou:.4f}")
        print(f"Val IoU   : {val_iou:.4f}")
        print(f"Val Acc   : {val_acc:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "resnet50_model.pth")
            print("🔥 Best model saved!")

    # Test
    model.load_state_dict(torch.load("resnet50_model.pth"))
    model.eval()

    test_iou = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            test_iou += iou_score(outputs, masks).item()

    print(f"\nFinal Test IoU: {test_iou / len(test_loader):.4f}")


# ==============================
# ENTRY POINT (CRITICAL)
# ==============================
if __name__ == "__main__":
    main()