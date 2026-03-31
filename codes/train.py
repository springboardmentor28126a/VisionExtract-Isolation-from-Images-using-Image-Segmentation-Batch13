import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

from dataset import COCOSegmentationDataset
from model import UNet
from metrics import iou_score, dice_score

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    # union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = COCOSegmentationDataset("train")
    val_dataset = COCOSegmentationDataset("val")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=2)

    model = UNet().to(device)
    
    # Load existing weights if available
    if os.path.exists("best_model_main_subject.pth"):
        model.load_state_dict(torch.load("best_model_main_subject.pth", map_location=device))
        print("Loaded saved model weights!")

    # criterion = nn.BCEWithLogitsLoss()
    pos_weight = torch.tensor([3.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_iou = 0
    epochs = 5

    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device).float()

            outputs = model(images)
            

            bce = criterion(outputs, masks)
            dice = dice_loss(outputs, masks)
            loss = 0.5 * bce + 0.5 * dice
            # loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_iou = 0
        val_dice = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device).float()
                outputs = model(images)

                val_iou += iou_score(outputs, masks).item()
                val_dice += dice_score(outputs, masks).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val IoU: {avg_val_iou:.4f}")
        print(f"Val Dice: {avg_val_dice:.4f}")
        print("-"*40)

        # 🔥 Save Best Model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), "best_model_main_subject.pth")
            print("✅ Best model saved!")


if __name__ == "__main__":
    main()