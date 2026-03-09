import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from model import UNet
from metrics import dice_score, iou_score

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load datasets
# -------------------------

train_dataset = SegmentationDataset(
    "data/train/images",
    "data/train/masks"
)

val_dataset = SegmentationDataset(
    "data/val/images",
    "data/val/masks"
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# -------------------------
# Model
# -------------------------

model = UNet().to(device)

loss_fn = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 15

best_val_accuracy = 0
best_epoch = 0

# -------------------------
# Training Loop
# -------------------------

for epoch in range(epochs):

    model.train()

    train_loss = 0
    train_accuracy = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)

        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()

        correct = (preds == masks).float().sum()
        accuracy = correct / torch.numel(preds)

        train_accuracy += accuracy.item()

    train_loss = train_loss / len(train_loader)
    train_accuracy = train_accuracy / len(train_loader)

    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_accuracy = 0
    val_dice = 0
    val_iou = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)

            preds_bin = torch.sigmoid(preds)
            preds_bin = (preds_bin > 0.5).float()

            correct = (preds_bin == masks).float().sum()
            accuracy = correct / torch.numel(preds_bin)

            val_accuracy += accuracy.item()

            val_dice += dice_score(preds, masks).item()
            val_iou += iou_score(preds, masks).item()

    val_accuracy = val_accuracy / len(val_loader)
    val_dice = val_dice / len(val_loader)
    val_iou = val_iou / len(val_loader)

    # -------------------------
    # Print results
    # -------------------------

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Dice: {val_dice:.4f}")
    print(f"Validation IoU: {val_iou:.4f}")
    print("--------------------------------------------------")

    # -------------------------
    # Save best model
    # -------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        torch.save(model.state_dict(), "best_model.pth")

# -------------------------
# Training Complete
# -------------------------

print("Training Finished")
print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")