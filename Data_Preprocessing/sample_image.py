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

# Get image info using filename
img_ids = coco.getImgIds()
img_id = None

for i in img_ids:
    info = coco.loadImgs(i)[0]
    if info['file_name'] == img_name:
        img_id = i
        img_info = info
        break

if img_id is None:
    print("Image not found in annotations!")
    exit()

# Read image
img_path = os.path.join(img_dir, img_name)
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get annotations
ann_ids = coco.getAnnIds(imgIds=img_id)
anns = coco.loadAnns(ann_ids)

# Copy image for overlay
overlay = image_rgb.copy()

# Create empty binary mask
binary_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

for ann in anns:
    # Bounding box
    x, y, w, h = map(int, ann['bbox'])
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Segmentation mask
    mask = coco.annToMask(ann)
    
    # Add to binary mask
    binary_mask = np.maximum(binary_mask, mask)
    
    # Color overlay (Red)
    overlay[mask == 1] = [255, 0, 0]

# ---- Save Outputs ----
output_overlay_path = os.path.join(img_dir, f"overlay_{img_name}")
output_mask_path = os.path.join(img_dir, f"mask_{img_name}")

cv2.imwrite(output_overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
cv2.imwrite(output_mask_path, binary_mask * 255)

print("Saved:")
print(output_overlay_path)
print(output_mask_path)
