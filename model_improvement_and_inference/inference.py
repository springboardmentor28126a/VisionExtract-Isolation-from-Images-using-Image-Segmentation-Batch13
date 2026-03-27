# =========================
# IMPORTS
# =========================
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50

# =========================
# PATH CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FOLDER = os.path.join(BASE_DIR, "week6_test_images")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "week6_outputs")

# 👉 Use FINAL trained model
MODEL_PATH = os.path.join(BASE_DIR, "deeplab_checkpoint.pth")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================
# SELECT IMAGE
# =========================
img_name = r"C:\Users\varsh\OneDrive\Desktop\VisionExtract_Segmentation\model_training\week6_test_images\image1.jpg"
img_path = os.path.join(INPUT_FOLDER, img_name)

if not os.path.exists(img_path):
    raise FileNotFoundError("❌ Image not found!")

# =========================
# LOAD IMAGE
# =========================
image = cv2.imread(img_path)

if image is None:
    raise ValueError("❌ Error reading image!")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]

# =========================
# LOAD MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = deeplabv3_resnet50(weights=None)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

# Load trained weights
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Handle both formats
if isinstance(checkpoint, dict) and 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model = model.to(device)
model.eval()

# =========================
# PREPROCESS IMAGE
# =========================
INPUT_SIZE = 256

image_resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
image_norm = image_resized / 255.0

image_tensor = torch.tensor(image_norm)\
    .permute(2, 0, 1)\
    .float()\
    .unsqueeze(0)\
    .to(device)

# =========================
# MODEL PREDICTION
# =========================
with torch.no_grad():
    output = model(image_tensor)['out']

    # Resize prediction back to original size
    output = F.interpolate(
        output,
        size=(h, w),
        mode='bilinear',
        align_corners=False
    )

    pred = torch.sigmoid(output[0, 0]).cpu().numpy()

# =========================
# POST-PROCESSING
# =========================

# Smooth prediction
pred = cv2.GaussianBlur(pred, (9, 9), 0)

# Adaptive threshold
threshold = 0.25 * pred.max()
binary_mask = (pred >= threshold).astype(np.uint8)

# Keep largest connected component
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

if num_labels > 1:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    binary_mask = (labels == largest_label).astype(np.uint8)

# Morphological cleaning
kernel = np.ones((5, 5), np.uint8)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

# =========================
# FINAL OUTPUT
# =========================
isolated = image * binary_mask[:, :, None]

# =========================
# SAVE OUTPUT
# =========================
output_path = os.path.join(OUTPUT_FOLDER, f"out_{img_name}")

cv2.imwrite(output_path, cv2.cvtColor(isolated, cv2.COLOR_RGB2BGR))

print(f"✅ Output saved at: {output_path}")

# =========================
# DISPLAY RESULTS
# =========================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(isolated)
plt.title("Background Removed")
plt.axis("off")

plt.tight_layout()
plt.show()