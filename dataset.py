import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as T

class CocoSubjectDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        # If no annotations, return empty mask
        if len(anns) == 0:
            mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        else:
            # Select largest object as main subject
            largest_ann = max(anns, key=lambda x: x['area'])
            mask = self.coco.annToMask(largest_ann)

        mask = Image.fromarray(mask * 255)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask