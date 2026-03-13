import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

from dataset import COCODataset
from model import get_model


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load dataset
dataset = COCODataset(
    image_dir="val2017",
    ann_file="annotations/instances_val2017.json"
)

# Reduce dataset size for faster training (first 100 images)
dataset = Subset(dataset, range(100))

loader = DataLoader(dataset, batch_size=2, shuffle=True)

print("Dataset loaded")
print("Total batches:", len(loader))


# Load model
model = get_model(num_classes=1)
model.to(device)


# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training epochs
epochs = 5

print("\nTraining started...\n")


# Training loop
for epoch in range(epochs):

    model.train()

    for batch_idx, (images, masks) in enumerate(loader):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")


print("\nTraining Finished")


# Save trained model
torch.save(model.state_dict(), "model.pth")

print("Model saved as model.pth")