# visualize_samples.py #
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import cv2
import numpy as np

from dataset.split_data import get_splits
from pycocotools.coco import COCO

# Load data
coco, train_ids, _, _ = get_splits()

for i in range(3):
    img_id = train_ids[i]
    img_info = coco.loadImgs(img_id)[0]

    # Load image
    img_path = os.path.join("coco2017/val2017", img_info['file_name'])
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create mask
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

    for ann in anns:
        if ann['category_id'] == 1:  # person only
            m = coco.annToMask(ann)
            mask = np.maximum(mask, m)

    # Plot
    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.title("Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Mask (Person)")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.show()