# explore_stats.py #
from pycocotools.coco import COCO
import os

# Paths
ann_file = "coco2017/annotations/instances_val2017.json"
img_dir = "coco2017/val2017"

# Load COCO
coco = COCO(ann_file)

# Stats
img_ids = coco.getImgIds()
ann_ids = coco.getAnnIds()
cats = coco.loadCats(coco.getCatIds())

print("===== DATASET INFO =====")
print(f"Total Images in Annotation File: {len(img_ids)}")
print(f"Total Images Present in Folder: {len(os.listdir(img_dir))}")
print(f"Total Annotations: {len(ann_ids)}")
print(f"Total Categories: {len(cats)}")

print("\nCategories:")
print([cat['name'] for cat in cats])