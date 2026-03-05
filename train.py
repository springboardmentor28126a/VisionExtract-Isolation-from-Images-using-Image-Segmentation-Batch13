from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from torch_dataset import SegmentationTorchDataset
from model import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Load dataset
full_dataset = SegmentationTorchDataset(
    image_dir="images/train",
    mask_dir="masks/train"
)

total_size = len(full_dataset)

train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size]
)

print(f"Total samples: {total_size}")
print(f"Train (70%): {train_size}")
print(f"Validation (15%): {val_size}")
print(f"Test (15%): {test_size}")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = UNet().to(DEVICE)

# CORRECT LOSS
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 5

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()
    train_loss = 0

    for i, (images, masks) in enumerate(train_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if i % 20 == 0:
            print(f"Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(images)
            loss = criterion(preds, masks)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# Test evaluation
model.eval()
test_loss = 0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        loss = criterion(preds, masks)
        test_loss += loss.item()

test_loss /= len(test_loader)

print(f"\nFinal Test Loss: {test_loss:.4f}")

torch.save(model.state_dict(), "unet_week4.pth")
print("Model saved as unet_week4.pth")