# validate_data.py #
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import cv2

from dataset.coco_dataset import COCOSegmentationDataset
from dataset.split_data import get_splits

# Load COCO
coco, train_ids, _, _ = get_splits()

# Dataset WITHOUT augmentation
dataset_no_aug = COCOSegmentationDataset(
    coco,
    train_ids[:5],
    "coco2017/val2017",
    augment=False
)

# Dataset WITH augmentation
dataset_aug = COCOSegmentationDataset(
    coco,
    train_ids[:5],
    "coco2017/val2017",
    augment=True
)

for i in range(3):

    # --- ORIGINAL IMAGE ---
    img_id = train_ids[i]
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join("coco2017/val2017", img_info['file_name'])

    original = cv2.imread(img_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # --- NO AUGMENTATION ---
    img_no_aug, mask = dataset_no_aug[i]
    img_no_aug = img_no_aug.permute(1, 2, 0).numpy()
    mask = mask.squeeze().numpy()

    # --- WITH AUGMENTATION ---
    img_aug, _ = dataset_aug[i]
    img_aug = img_aug.permute(1, 2, 0).numpy()

    # --- PLOT ---
    plt.figure(figsize=(16,4))

    plt.subplot(1,4,1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.title("Preprocessed (No Aug)")
    plt.imshow(img_no_aug)
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.title("Augmented")
    plt.imshow(img_aug)
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.show()