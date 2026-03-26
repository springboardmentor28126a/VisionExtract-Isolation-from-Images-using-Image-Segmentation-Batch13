import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCODataset(Dataset):
    def __init__(self, image_dir, ann_file):
        self.image_dir = image_dir
        self.coco = COCO(ann_file)

        # 🔥 LIMIT DATA FOR SPEED
        self.ids = list(self.coco.imgs.keys())[:300]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(img_path)

        if image is None:
            image = np.zeros((256,256,3), dtype=np.uint8)

        image = cv2.resize(image, (256,256))

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((256,256), dtype=np.uint8)

        for ann in anns:
            m = self.coco.annToMask(ann)
            m = cv2.resize(m, (256,256))
            mask = np.maximum(mask, m)

        image = image / 255.0

        image = torch.tensor(image).permute(2,0,1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask