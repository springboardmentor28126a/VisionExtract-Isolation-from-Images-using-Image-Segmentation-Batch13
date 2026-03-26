import torch
from torch.utils.data import DataLoader
from dataset import COCODataset
from model import get_model
import torch.nn as nn
import torch.optim as optim

device = torch.device("cpu")
print("Using device:", device)

dataset = COCODataset(
    image_dir="val2017",
    ann_file="annotations/instances_val2017.json"
)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = get_model().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10

for epoch in range(EPOCHS):
    total_loss = 0

    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} Completed, Total Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "model.pth")
print("✅ Model saved successfully!")