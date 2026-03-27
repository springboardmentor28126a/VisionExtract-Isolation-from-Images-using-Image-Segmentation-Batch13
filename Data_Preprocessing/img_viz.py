# import cv2
# import matplotlib.pyplot as plt
# from pycocotools.coco import COCO
# import os
# import numpy as np

# # ---- Paths ----
# ann_file = r"E:\Infosys_Springboard_Internship\COCO_Dataset(25 GB)\coco2017\annotations\instances_train2017.json"
# img_dir = r"E:\Infosys_Springboard_Internship\COCO_Dataset(25 GB)\coco2017\train2017"
# img_name = "000000003867.jpg"

# # Load COCO
# coco = COCO(ann_file)

# # Find image id using filename
# img_ids = coco.getImgIds()
# img_id = None

# for i in img_ids:
#     info = coco.loadImgs(i)[0]
#     if info['file_name'] == img_name:
#         img_id = i
#         img_info = info
#         break

# if img_id is None:
#     print("Image not found in annotations")
#     exit()

# # Read image
# img_path = os.path.join(img_dir, img_name)
# image = cv2.imread(img_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Get annotations
# ann_ids = coco.getAnnIds(imgIds=img_id)
# anns = coco.loadAnns(ann_ids)

# # Create overlay image
# overlay = image_rgb.copy()

# # Create binary mask
# binary_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

# for ann in anns:
#     mask = coco.annToMask(ann)
#     binary_mask = np.maximum(binary_mask, mask)
#     overlay[mask == 1] = [255, 0, 0]   # red mask

# plt.figure(figsize=(15,5))

# # 1. Original
# plt.subplot(1,3,1)
# plt.imshow(image_rgb)
# plt.title("Original Image")
# plt.axis("off")

# # 2. Segmented Overlay
# plt.subplot(1,3,2)
# plt.imshow(overlay)
# plt.title("Segmented Overlay")
# plt.axis("off")

# # 3. Binary Mask
# plt.subplot(1,3,3)
# plt.imshow(binary_mask, cmap='gray')
# plt.title("Binary Mask")
# plt.axis("off")

# plt.tight_layout()
# plt.show()
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

# ================================
# PATHS
# ================================
ann_file = r"E:\COCO DATASET\COCO_Dataset(25 GB)\coco2017\annotations\instances_val2017.json"
img_dir  = r"E:\COCO DATASET\COCO_Dataset(25 GB)\coco2017\val2017"

# OUTPUT DIRS
out_img_dir  = r"E:\COCO Dataset\processed_binary\images"
out_mask_dir = r"E:\COCO Dataset\processed_binary\masks"

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

# ================================
# LOAD COCO
# ================================
coco = COCO(ann_file)
img_ids = coco.getImgIds()

print("Total images:", len(img_ids))

# ================================
# PROCESS ALL IMAGES
# ================================
processed = 0

for img_id in tqdm(img_ids):

    # Load image info
    img_info = coco.loadImgs(img_id)[0]
    img_name = img_info['file_name']

    img_path = os.path.join(img_dir, img_name)

    # Read image
    image = cv2.imread(img_path)

    if image is None:
        continue

    h, w = img_info['height'], img_info['width']

    # Get annotations
    ann_ids = coco.getAnnIds(imgIds=img_id)

    # ❗ Skip images with no objects
    if len(ann_ids) == 0:
        continue

    anns = coco.loadAnns(ann_ids)

    # ================================
    # CREATE BINARY MASK
    # ================================
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    for ann in anns:
        mask = coco.annToMask(ann)

        # Combine all objects into one mask
        binary_mask = np.maximum(binary_mask, mask)

    # Convert to 0 or 255
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    # ================================
    # SAVE
    # ================================
    cv2.imwrite(os.path.join(out_img_dir, img_name), image)
    cv2.imwrite(os.path.join(out_mask_dir, img_name), binary_mask)

    processed += 1

print(f"\n✅ Done! Processed {processed} images.")