import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class COCODataset(Dataset):
    def __init__(self, annotation_file, image_folder, image_size=512, transform=False):
        """
        annotation_file : path to COCO annotation json
        image_folder    : path to val2017 or train2017 folder
        image_size      : resize dimension (default 512x512)
        transform       : whether to apply augmentation
        """
        self.coco = COCO(annotation_file)
        self.image_folder = image_folder
        self.image_ids = self.coco.getImgIds()
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # Load image info
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.image_folder, img_info['file_name'])

        # =========================
        # STEP 1 — Load Image
        # =========================
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # =========================
        # STEP 2 — Load Mask
        # =========================
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))

        # =========================
        # STEP 3 — Convert to Binary
        # =========================
        binary_mask = (mask > 0).astype(np.uint8)

        # =========================
        # STEP 4 — Resize
        # =========================
        image = cv2.resize(image, (self.image_size, self.image_size))
        binary_mask = cv2.resize(
            binary_mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        # =========================
        # STEP 5 — Normalize Image
        # =========================
        image = image / 255.0

        # =========================
        # STEP 6 — Data Augmentation (Optional)
        # =========================
        if self.transform:
            if np.random.rand() > 0.5:
                image = cv2.flip(image, 1)
                binary_mask = cv2.flip(binary_mask, 1)

        # =========================
        # STEP 7 — Convert to Tensor
        # =========================
        image = torch.tensor(image).permute(2, 0, 1).float()
        binary_mask = torch.tensor(binary_mask).unsqueeze(0).float()

        return image, binary_mask
