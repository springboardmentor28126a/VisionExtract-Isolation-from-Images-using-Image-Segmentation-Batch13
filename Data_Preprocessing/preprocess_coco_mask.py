import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import os

# ---------------- PATHS ----------------
img_name = "000000000025.jpg"

img_dir = r"E:\Infosys_Springboard_Internship\COCO_Dataset(25 GB)\coco2017\train2017"
ann_file = r"E:\Infosys_Springboard_Internship\COCO_Dataset(25 GB)\coco2017\annotations\instances_train2017.json"

target_size = (512, 512)

# ---------------- LOAD COCO ----------------
coco = COCO(ann_file)

# Find image_id using filename
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

# ---------------- READ IMAGE ----------------
img_path = os.path.join(img_dir, img_name)
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------------- PREPROCESSING ----------------

# Resize
resized = cv2.resize(image_rgb, target_size)

# Denoise
denoised = cv2.GaussianBlur(resized, (5, 5), 0)

# CLAHE
lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l_clahe = clahe.apply(l)

lab_clahe = cv2.merge((l_clahe, a, b))
enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

# Normalize
preprocessed_image = enhanced / 255.0

# ---------------- GENERATE COCO MASK ----------------
ann_ids = coco.getAnnIds(imgIds=img_id)
anns = coco.loadAnns(ann_ids)

# Create empty mask (original size)
mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

for ann in anns:
    m = coco.annToMask(ann)
    mask = np.maximum(mask, m)

# ---------------- RESIZE MASK ----------------
mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

# Ensure binary
mask_resized = (mask_resized > 0).astype(np.uint8)

# ---------------- DISPLAY ----------------
plt.figure(figsize=(15,5))

# Original
plt.subplot(1,3,1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

# Preprocessed
plt.subplot(1,3,2)
plt.imshow(preprocessed_image)
plt.title("Preprocessed (512x512)")
plt.axis("off")

# Mask
plt.subplot(1,3,3)
plt.imshow(mask_resized, cmap='gray')
plt.title("COCO Mask (512x512)")
plt.axis("off")

plt.tight_layout()
plt.show()
