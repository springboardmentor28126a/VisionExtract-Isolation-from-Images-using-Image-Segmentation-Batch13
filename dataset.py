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
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.image_dir, img_info['file_name'])

        image = cv2.imread(img_path)
        image = cv2.resize(image, (256,256))
        image = image / 255.0
        image = torch.tensor(image).permute(2,0,1).float()

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((256,256))

        for ann in anns:
            m = self.coco.annToMask(ann)
            m = cv2.resize(m,(256,256))
            mask = np.maximum(mask,m)

        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask