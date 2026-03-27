import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

from model import UNet
from torch_dataset import SegmentationTorchDataset


assert torch.cuda.is_available(), "CUDA GPU is required but not available."
DEVICE = torch.device("cuda")

# For de-normalizing ImageNet-normalized tensors for display
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)

THRESHOLD = 0.4


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
 

    if mask.max() == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask * 255).astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return mask

    # stats[0] is background; pick component with largest area
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    return (labels == largest_idx).astype(np.uint8)


# ---------------- LOAD DATASET ----------------
dataset = SegmentationTorchDataset(
    image_dir="images/train",
    mask_dir="masks/train"
)


model = UNet().to(DEVICE)
try:
    state_dict = torch.load("best_unet.pth", map_location="cuda", weights_only=True)
except TypeError:
    # Older PyTorch: weights_only not supported
    state_dict = torch.load("best_unet.pth", map_location="cuda")
model.load_state_dict(state_dict)
model.eval()


# ---------------- RANDOM SAMPLE ----------------
idx = random.randint(0, len(dataset) - 1)

image, mask = dataset[idx]
image = image.to(DEVICE)
mask_np = mask.squeeze(0).detach().cpu().numpy().astype(np.uint8)


# ---------------- MODEL PREDICTION ----------------
with torch.no_grad():

    pred = model(image.unsqueeze(0))

    pred = torch.sigmoid(pred).squeeze().detach().cpu().numpy()


# ---------------- THRESHOLD ----------------
pred = (pred > THRESHOLD).astype(np.uint8)



kernel = np.ones((3, 3), np.uint8)
pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
pred = keep_largest_component(pred)


# ---------------- PREPARE IMAGE ----------------
image_vis = image.detach().cpu().permute(1, 2, 0)
image_vis = (image_vis * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
image_np = image_vis.numpy()


# ---------------- EXTRACT OBJECT ----------------
extracted = image_np * pred[:,:,None]
extracted_gt = image_np * mask_np[:, :, None]


# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(16,4))


# Original Image
plt.subplot(1,4,1)
plt.title("Original Image")
plt.imshow(image_np)
plt.axis("off")


# Ground Truth Mask
plt.subplot(1,4,2)
plt.title("Ground Truth Mask")
plt.imshow(mask_np, cmap="gray")
plt.axis("off")


# Predicted Mask
plt.subplot(1,4,3)
plt.title("Predicted Mask")
plt.imshow(pred, cmap="gray")
plt.axis("off")


# Extracted Object (Pred)
plt.subplot(1,4,4)
plt.title("Extracted Object (Pred)")
plt.imshow(extracted)
plt.axis("off")


plt.tight_layout()
plt.show()
