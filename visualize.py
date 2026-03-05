import torch
import matplotlib.pyplot as plt
import random
from model import UNet
from torch_dataset import SegmentationTorchDataset

dataset = SegmentationTorchDataset(
    image_dir="images/train",
    mask_dir="masks/train"
)

model = UNet()
model.load_state_dict(torch.load("unet_week4.pth", map_location="cpu"))
model.eval()

idx = random.randint(0, len(dataset) - 1)
image, mask = dataset[idx]

with torch.no_grad():
    pred = model(image.unsqueeze(0)).squeeze().numpy()

plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(image.permute(1, 2, 0))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Ground Truth")
plt.imshow(mask.squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow(pred, cmap="gray")
plt.colorbar()
plt.axis("off")

plt.show()