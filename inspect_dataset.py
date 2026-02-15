import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from preprocess import preprocess_image_and_mask, apply_mask

ann_file = "dataset/annotations/instances_val2017.json"
images_path = "dataset/images/val2017"

coco = COCO(ann_file)

img_ids = coco.getImgIds()
img_info = coco.loadImgs(img_ids[21])[0]

img_path = os.path.join(images_path, img_info['file_name'])
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate binary mask (all objects)
ann_ids = coco.getAnnIds(imgIds=img_info['id'])
anns = coco.loadAnns(ann_ids)

mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

for ann in anns:
    obj_mask = coco.annToMask(ann)
    mask = np.maximum(mask, obj_mask)

# Preprocess
image_resized, image_normalized, mask_resized = preprocess_image_and_mask(image, mask)

# Apply mask
isolated_image = apply_mask(image_resized, mask_resized)

# Display
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Binary Mask")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(image_resized)
plt.title("Resized Image")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(isolated_image)
plt.title("Subject Isolated Output")
plt.axis("off")

plt.tight_layout()
plt.show()
