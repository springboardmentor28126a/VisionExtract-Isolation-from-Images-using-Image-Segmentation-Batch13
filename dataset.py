import os
import random
import numpy as np
import cv2
from preprocess import preprocess_image


class SegmentationDataset:
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment

        images = sorted(os.listdir(image_dir))
        masks = set(os.listdir(mask_dir))

        # ✅ keep only images that have corresponding masks
        self.images = [
            img for img in images
            if img.replace(".jpg", ".png") in masks
        ]

        print(f"✅ Valid image-mask pairs: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]

        image_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = preprocess_image(image_path)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        mask = np.where(mask > 127, 1, 0).astype(np.uint8)

        if self.augment and random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        return image, mask
