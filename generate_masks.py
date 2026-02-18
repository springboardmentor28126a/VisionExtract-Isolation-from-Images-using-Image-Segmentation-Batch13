import os
import numpy as np
import cv2
from pycocotools.coco import COCO

# paths
ANNOTATION_FILE = "annotations/instances_val2017.json"
IMAGE_DIR = "images/train"
MASK_DIR = "masks/train"

os.makedirs(MASK_DIR, exist_ok=True)

coco = COCO(ANNOTATION_FILE)

img_ids = coco.getImgIds()

print("Total images in annotation:", len(img_ids))

for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info["file_name"]

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    if len(anns) == 0:
        continue

    height = img_info["height"]
    width = img_info["width"]

    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in anns:
        ann_mask = coco.annToMask(ann)
        mask = np.maximum(mask, ann_mask)

    # convert to 0 / 255
    mask = (mask * 255).astype(np.uint8)

    mask_path = os.path.join(MASK_DIR, file_name.replace(".jpg", ".png"))
    cv2.imwrite(mask_path, mask)

print("✅ Masks generated successfully")
