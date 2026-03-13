import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

from model import UNet
from dataset import COCOSegmentationDataset
from preprocess_test2 import preprocess_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)
model.load_state_dict(torch.load("best_model_main_subject.pth"))
model.eval()

val_dataset = COCOSegmentationDataset("val")

def keep_largest_component(mask):
    # Convert mask to uint8
    mask = mask.astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Ignore background (label 0)
    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]

    for i in range(2, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > largest_area:
            largest_area = stats[i, cv2.CC_STAT_AREA]
            largest_label = i

    # Create cleaned mask
    cleaned_mask = np.zeros_like(mask)
    cleaned_mask[labels == largest_label] = 1

    return cleaned_mask

# Take first sample
image, mask = val_dataset[288]

image_np = image.permute(1,2,0).numpy()

with torch.no_grad():
    input_tensor = image.unsqueeze(0).to(device)
    output = model(input_tensor)
    pred = torch.sigmoid(output)
    pred = (pred > 0.5).float()

pred_mask = pred.squeeze().cpu().numpy()

pred_mask = keep_largest_component(pred_mask)

kernel = np.ones((5,5), np.uint8)
pred_mask = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
pred_mask = cv2.medianBlur(pred_mask, 5)

# Create isolated image
isolated = image_np * np.expand_dims(pred_mask, axis=-1)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image_np)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(pred_mask, cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(isolated)
plt.title("Isolated Subject")
plt.axis("off")

plt.show()