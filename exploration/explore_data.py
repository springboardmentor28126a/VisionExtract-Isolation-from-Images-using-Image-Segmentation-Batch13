# explore_data.py #
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from dataset.coco_dataset import COCOSegmentationDataset
from dataset.split_data import get_splits

# Get splits
coco, train_ids, val_ids, test_ids = get_splits()

# Create dataset
dataset = COCOSegmentationDataset(
    coco,
    train_ids,
    "coco2017/val2017"
)

# Get one sample
image, mask = dataset[0]

# Convert for display
image = image.permute(1, 2, 0).numpy()
mask = mask.squeeze().numpy()

# Show
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Image")

plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title("Mask")

plt.show()