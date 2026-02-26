import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import CocoSubjectDataset
from model_unet import UNet

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Transform (Reduced Size for CPU)
# -----------------------------
transform = T.Compose([
    T.Resize((128, 128)),   # Smaller image = faster training
    T.ToTensor(),
])

# -----------------------------
# Dataset
# -----------------------------
dataset = CocoSubjectDataset(
    image_dir="data/data/coco2017/train2017",
    annotation_file="data/data/coco2017/annotations/instances_train2017.json",
    transform=transform
)

print("Total images in dataset:", len(dataset))

# -----------------------------
# Use Small Subset (Fast Debug)
# -----------------------------
subset_size = 50
dataset = torch.utils.data.Subset(dataset, range(subset_size))

print("Using subset size:", subset_size)

# -----------------------------
# DataLoader
# -----------------------------
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# -----------------------------
# Model
# -----------------------------
model = UNet().to(device)

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# Training Loop
# -----------------------------
num_epochs = 1

for epoch in range(num_epochs):
    print(f"\nStarting Epoch {epoch+1}/{num_epochs}")
    model.train()
    epoch_loss = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device).float()

        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")

    print(f"\nEpoch Loss: {epoch_loss / len(loader):.4f}")

print("\nTraining Completed Successfully ")




torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth ")