import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import COCODataset
from model import get_model

device = torch.device("cpu")

# Load dataset
dataset = COCODataset("val2017", "annotations/instances_val2017.json")

image, _ = dataset[0]

# Load model
model = get_model()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Predict mask
with torch.no_grad():
    output = model(image.unsqueeze(0))
    pred = torch.sigmoid(output)
    mask = (pred > 0.5).float()

# Convert tensors
image_np = image.permute(1, 2, 0).numpy()
mask_np = mask.squeeze().numpy()

# Apply mask (isolation)
isolated = image_np * np.expand_dims(mask_np, axis=2)

# Plot results
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(image_np)

plt.subplot(1,3,2)
plt.title("Predicted Mask")
plt.imshow(mask_np, cmap='gray')

plt.subplot(1,3,3)
plt.title("Isolated Object")
plt.imshow(isolated)

plt.show()