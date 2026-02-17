import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

# Paths
annotation_file = "annotations/instances_val2017.json"
image_folder = "val2017"

# Load COCO
coco = COCO(annotation_file)

# Get first image
img_ids = coco.getImgIds()
img_id = img_ids[0]

img_info = coco.loadImgs(img_id)[0]
image_path = os.path.join(image_folder, img_info['file_name'])

# Read image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get annotations
ann_ids = coco.getAnnIds(imgIds=img_id)
anns = coco.loadAnns(ann_ids)

# Create empty mask
final_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

# First find maximum area
areas = [np.sum(coco.annToMask(ann)) for ann in anns]
max_area = max(areas)

# Keep objects that are at least 30% of largest object
threshold = 0.3 * max_area

for ann in anns:
    mask = coco.annToMask(ann)
    area = np.sum(mask)

    if area >= threshold:
        final_mask = np.maximum(final_mask, mask)

# Convert to binary
binary_mask = (final_mask > 0).astype(np.uint8)

# Show only 2 images
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Main Objects Mask")
plt.imshow(binary_mask, cmap="gray")
plt.axis("off")

plt.show()
