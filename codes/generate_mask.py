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
cat_ids = coco.getCatIds(catNms=['person'])
print("Person Category ID:", cat_ids)

def generate_person_mask(image_id):
    img_info = coco.loadImgs(image_id)[0]
    height = img_info['height']
    width = img_info['width']

    ann_ids = coco.getAnnIds(imgIds=image_id, catIds=cat_ids)
    anns = coco.loadAnns(ann_ids)

    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in anns:
        ann_mask = coco.annToMask(ann)
        mask = np.maximum(mask, ann_mask)

    return mask, img_info['file_name']
