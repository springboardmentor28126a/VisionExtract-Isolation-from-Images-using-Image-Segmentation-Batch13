import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from preprocess import preprocess_image

class SegmentationTorchDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        images = sorted(os.listdir(image_dir))
        masks = set(os.listdir(mask_dir))

        self.images = [
            img for img in images
            if img.replace(".jpg", ".png") in masks
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(
            self.mask_dir, img_name.replace(".jpg", ".png")
        )

        image = preprocess_image(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 127).astype(np.float32)

        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask