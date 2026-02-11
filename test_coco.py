import os
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# Paths
image_folder = "val2017"
annotation_file = "annotations/instances_val2017.json"

# Load COCO annotations
coco = COCO(annotation_file)

# Get first image ID
image_ids = coco.getImgIds()
img_id = image_ids[0]

# Load image info
img_info = coco.loadImgs(img_id)[0]
image_path = os.path.join(image_folder, img_info['file_name'])

# Read image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get annotations for this image
ann_ids = coco.getAnnIds(imgIds=img_id)
anns = coco.loadAnns(ann_ids)

# Display image
plt.imshow(image)
plt.axis("off")

# Draw segmentation masks
coco.showAnns(anns)

plt.show()
