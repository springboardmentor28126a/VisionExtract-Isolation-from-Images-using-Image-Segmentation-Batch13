import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

from dataset import CocoSubjectDataset

# ----------------------------
# Transform (same for image & mask)
# ----------------------------
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# ----------------------------
# Dataset
# ----------------------------
dataset = CocoSubjectDataset(
    image_dir="data/data/coco2017/train2017",
    annotation_file="data/data/coco2017/annotations/instances_train2017.json",
    transform=transform
)

print("Total images:", len(dataset))

# ----------------------------
# DataLoader
# ----------------------------
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

# ----------------------------
# Fetch one sample
# ----------------------------
image, mask = next(iter(loader))

print("Image shape:", image.shape)   # [1, 3, 256, 256]
print("Mask shape:", mask.shape)     # [1, 1, 256, 256]

# ----------------------------
# Visualization
# ----------------------------
img = image[0].permute(1, 2, 0)
msk = mask[0].squeeze()

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(msk, cmap="gray")
plt.title("Subject Mask")
plt.axis("off")

plt.show()