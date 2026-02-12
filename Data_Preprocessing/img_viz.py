import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import os
import numpy as np

# ---- Paths ----
ann_file = r"E:\Infosys_Springboard_Internship\COCO_Dataset(25 GB)\coco2017\annotations\instances_train2017.json"
img_dir = r"E:\Infosys_Springboard_Internship\COCO_Dataset(25 GB)\coco2017\train2017"
img_name = "000000003867.jpg"

# Load COCO
coco = COCO(ann_file)

# Find image id using filename
img_ids = coco.getImgIds()
img_id = None

for i in img_ids:
    info = coco.loadImgs(i)[0]
    if info['file_name'] == img_name:
        img_id = i
        img_info = info
        break

if img_id is None:
    print("Image not found in annotations")
    exit()

# Read image
img_path = os.path.join(img_dir, img_name)
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get annotations
ann_ids = coco.getAnnIds(imgIds=img_id)
anns = coco.loadAnns(ann_ids)

# Create overlay image
overlay = image_rgb.copy()

# Create binary mask
binary_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

for ann in anns:
    mask = coco.annToMask(ann)
    binary_mask = np.maximum(binary_mask, mask)
    overlay[mask == 1] = [255, 0, 0]   # red mask

plt.figure(figsize=(15,5))

# 1. Original
plt.subplot(1,3,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

# 2. Segmented Overlay
plt.subplot(1,3,2)
plt.imshow(overlay)
plt.title("Segmented Overlay")
plt.axis("off")

# 3. Binary Mask
plt.subplot(1,3,3)
plt.imshow(binary_mask, cmap='gray')
plt.title("Binary Mask")
plt.axis("off")

plt.tight_layout()
plt.show()
