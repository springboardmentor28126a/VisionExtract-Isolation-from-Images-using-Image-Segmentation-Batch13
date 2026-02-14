import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# --------------------------------
# 1️⃣ DATASET PATH (CHANGE ONLY IF NEEDED)
# --------------------------------
DATASET_PATH = r"E:\Datasets\COCO_FINAL"
ANNOTATION_FILE = os.path.join(DATASET_PATH, "annotations", "instances_train2017.json")

TARGET_SIZE = (512, 512)

print("Checking dataset path...")

if not os.path.exists(DATASET_PATH):
    print("Dataset folder NOT found ❌")
    exit()

if not os.path.exists(ANNOTATION_FILE):
    print("Annotation file NOT found ❌")
    exit()

print("Dataset Found ✅")
print("Loading COCO annotations...")

# --------------------------------
# 2️⃣ LOAD COCO
# --------------------------------
coco = COCO(ANNOTATION_FILE)

# --------------------------------
# 3️⃣ GET FIRST IMAGE
# --------------------------------
img_ids = coco.getImgIds()
img_id = img_ids[0]

img_info = coco.loadImgs(img_id)[0]
image_path = os.path.join(DATASET_PATH, "train2017", img_info['file_name'])

print("Loading Image:", img_info['file_name'])

# --------------------------------
# 4️⃣ READ IMAGE
# --------------------------------
image = cv2.imread(image_path)

if image is None:
    print("Image not found ❌")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image (model-ready size)
image = cv2.resize(image, TARGET_SIZE)

# --------------------------------
# 5️⃣ CREATE MASK
# --------------------------------
ann_ids = coco.getAnnIds(imgIds=img_id)
anns = coco.loadAnns(ann_ids)

mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

for ann in anns:
    mask = np.maximum(mask, coco.annToMask(ann))

# Resize mask with NEAREST (important)
mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

# Ensure binary
mask = (mask > 0).astype(np.uint8)

print("Mask created ✅")

# --------------------------------
# 6️⃣ APPLY MASK (REMOVE BACKGROUND)
# --------------------------------
isolated = image * mask[:, :, np.newaxis]

# --------------------------------
# 7️⃣ SAVE OUTPUTS
# --------------------------------
os.makedirs("outputs", exist_ok=True)

cv2.imwrite("outputs/original.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.imwrite("outputs/mask.png", mask * 255)
cv2.imwrite("outputs/isolated.png", cv2.cvtColor(isolated, cv2.COLOR_RGB2BGR))

print("Outputs saved in /outputs folder ✅")

# --------------------------------
# 8️⃣ SHOW RESULTS
# --------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Segmentation Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Background Removed")
plt.imshow(isolated)
plt.axis("off")

plt.tight_layout()
plt.show()
