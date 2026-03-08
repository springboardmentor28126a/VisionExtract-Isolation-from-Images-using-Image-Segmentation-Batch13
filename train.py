import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import COCODataset
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = COCODataset(
    image_dir="val2017",
    ann_file="annotations/instances_val2017.json"
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

model = get_model(num_classes=1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

epochs = 5

for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images, masks in dataloader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(),"model.pth")

print("Training Finished")