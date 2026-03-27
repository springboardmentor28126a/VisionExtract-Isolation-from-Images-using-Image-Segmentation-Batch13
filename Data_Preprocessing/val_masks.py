# import cv2
# import numpy as np
# import os
# from pycocotools.coco import COCO
# from tqdm import tqdm

# # ---------------- PATHS ----------------
# img_dir = r"E:\Infosys_Springboard_Internship\COCO_Dataset(25 GB)\coco2017\val2017"
# ann_file = r"E:\Infosys_Springboard_Internship\COCO_Dataset(25 GB)\coco2017\annotations\instances_val2017.json"

# output_dir = r"E:\Infosys_Springboard_Internship\processed_val"
# mask_dir = os.path.join(output_dir, "masks")
# masked_img_dir = os.path.join(output_dir, "masked_images")

# os.makedirs(mask_dir, exist_ok=True)
# os.makedirs(masked_img_dir, exist_ok=True)

# target_size = (512, 512)
# max_images = 5000   # process only 5000 images

# # ---------------- LOAD COCO ----------------
# coco = COCO(ann_file)
# img_ids = coco.getImgIds()[:max_images]

# print("Total images to process:", len(img_ids))

# # ---------------- PROCESS LOOP ----------------
# for img_id in tqdm(img_ids):

#     img_info = coco.loadImgs(img_id)[0]
#     img_name = img_info['file_name']
#     img_path = os.path.join(img_dir, img_name)

#     # Read image
#     image = cv2.imread(img_path)
#     if image is None:
#         continue

#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Resize image
#     image_resized = cv2.resize(image_rgb, target_size)

#     # --------- Generate Mask ----------
#     ann_ids = coco.getAnnIds(imgIds=img_id)
#     anns = coco.loadAnns(ann_ids)

#     mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

#     for ann in anns:
#         m = coco.annToMask(ann)
#         mask = np.maximum(mask, m)

#     # Resize mask
#     mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
#     mask_resized = (mask_resized > 0).astype(np.uint8)

#     # --------- Create Masked Image ----------
#     masked_image = image_resized.copy()
#     masked_image[mask_resized == 0] = 0

#     # --------- Save ----------
#     mask_path = os.path.join(mask_dir, img_name)
#     masked_img_path = os.path.join(masked_img_dir, img_name)

#     cv2.imwrite(mask_path, mask_resized * 255)
#     cv2.imwrite(masked_img_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

# print("Processing completed!")
import cv2
import numpy as np
import os
from pycocotools.coco import COCO
from tqdm import tqdm

# ---------------- PATHS ----------------
img_dir = r"E:\COCO DATASET\COCO_Dataset(25 GB)\coco2017\val2017"
ann_file = r"E:\COCO DATASET\COCO_Dataset(25 GB)\coco2017\annotations\instances_val2017.json"

output_dir = r"E:\COCO DATASET\processed_binary"
mask_dir = os.path.join(output_dir, "masks")
masked_img_dir = os.path.join(output_dir, "images")

os.makedirs(mask_dir, exist_ok=True)
os.makedirs(masked_img_dir, exist_ok=True)

target_size = (512, 512)
max_images = 5000

# ---------------- LOAD COCO ----------------
coco = COCO(ann_file)
img_ids = coco.getImgIds()[:max_images]

print("Total images to process:", len(img_ids))

# ---------------- STATS ----------------
skipped_empty = 0
skipped_small = 0
processed = 0

# ---------------- PROCESS LOOP ----------------
for img_id in tqdm(img_ids):

    img_info = coco.loadImgs(img_id)[0]
    img_name = img_info['file_name']
    img_path = os.path.join(img_dir, img_name)

    # Read image
    image = cv2.imread(img_path)
    if image is None:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, target_size)

    # ---------------- MASK GENERATION ----------------
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # ❌ Skip empty annotations
    if len(anns) == 0:
        skipped_empty += 1
        continue

    # ✅ Select largest object
    largest_area = 0
    best_mask = None

    for ann in anns:
        m = coco.annToMask(ann)
        area = m.sum()

        if area > largest_area:
            largest_area = area
            best_mask = m

    if best_mask is None:
        skipped_empty += 1
        continue

    # Resize mask
    mask_resized = cv2.resize(best_mask, target_size, interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized > 0).astype(np.uint8)

    # ❌ Skip very small objects
    if mask_resized.sum() < 500:
        skipped_small += 1
        continue

    # ---------------- MORPHOLOGICAL CLEANING ----------------
    kernel = np.ones((5,5), np.uint8)

    mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)
    mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)

    # ---------------- BALANCE CHECK ----------------
    fg_ratio = mask_resized.sum() / mask_resized.size

    # ❌ Skip if too much background
    if fg_ratio < 0.01:
        skipped_small += 1
        continue

    # ---------------- CREATE MASKED IMAGE ----------------
    masked_image = image_resized.copy()
    masked_image[mask_resized == 0] = 0

    # ---------------- SAVE ----------------
    mask_path = os.path.join(mask_dir, img_name)
    masked_img_path = os.path.join(masked_img_dir, img_name)

    cv2.imwrite(mask_path, mask_resized * 255)
    cv2.imwrite(masked_img_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

    processed += 1

# ---------------- SUMMARY ----------------
print("\nProcessing completed!")
print(f"Processed images : {processed}")
print(f"Skipped empty    : {skipped_empty}")
print(f"Skipped small    : {skipped_small}")