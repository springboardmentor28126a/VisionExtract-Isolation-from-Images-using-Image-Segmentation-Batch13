from dataset import SegmentationDataset
import matplotlib.pyplot as plt
import random

IMAGE_DIR = "images/train"
MASK_DIR = "masks/train"

dataset = SegmentationDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    augment=False
)

NUM_SAMPLES = 6  # must be even
indices = random.sample(range(len(dataset)), NUM_SAMPLES)

plt.figure(figsize=(12, 8))

for i, idx in enumerate(indices):
    image, mask = dataset[idx]

    plt.subplot(NUM_SAMPLES // 2, 4, i * 2 + 1)
    plt.imshow(image)
    plt.title(f"Image {idx}", fontsize=10)
    plt.axis("off")

    plt.subplot(NUM_SAMPLES // 2, 4, i * 2 + 2)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Mask {idx}", fontsize=10)
    plt.axis("off")

plt.suptitle(
    "Random Image–Mask Samples (Week 2 Validation)",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
