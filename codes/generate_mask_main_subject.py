import numpy as np
import os
from pycocotools.coco import COCO

# Correct paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

ANNOTATION_FILE = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")
IMAGE_DIR = os.path.join(DATA_DIR, "val2017")

coco = COCO(ANNOTATION_FILE)

# Get category ID for "person"
# cat_ids = coco.getCatIds(catNms=['person'])
# cat_ids = coco.getCatIds()
# print("Person Category ID:", cat_ids)

def generate_full_mask(image_id):
    img_info = coco.loadImgs(image_id)[0]
    height = img_info['height']
    width = img_info['width']

    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    largest_area = 0
    best_mask = np.zeros((height, width), dtype=np.uint8)

    for ann in anns:
        ann_mask = coco.annToMask(ann)
        area = ann_mask.sum()

        if area > largest_area:
            largest_area = area
            best_mask = ann_mask

    return best_mask, img_info['file_name']
