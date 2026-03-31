import os
import cv2
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))   # ✅ FIX (important)
        self.size = size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_name = self.images[index]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize (VERY IMPORTANT for model consistency)
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size)

        # Normalize
        image = image.astype("float32") / 255.0
        mask = mask.astype("float32") / 255.0

        # Convert to tensor
        image = torch.tensor(image).permute(2, 0, 1)   # (C, H, W)
        mask = torch.tensor(mask).unsqueeze(0)         # (1, H, W)

        return image, mask