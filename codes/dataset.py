import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from preprocessing import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

ANNOTATION_FILE = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")
IMAGE_DIR = os.path.join(DATA_DIR, "val2017")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")

coco = COCO(ANNOTATION_FILE)

class COCOSegmentationDataset(Dataset):

    def __init__(self, split="train"):
        split_file = os.path.join(SPLIT_DIR, f"{split}_ids.txt")

        with open(split_file, "r") as f:
            self.image_ids = [int(line.strip()) for line in f]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        img_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(IMAGE_DIR, img_info['file_name'])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate full mask (all classes)
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)

        # mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        # for ann in anns:
        #     ann_mask = coco.annToMask(ann)
        #     mask = np.maximum(mask, ann_mask)

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        largest_area = 0
        largest_ann = None

        for ann in anns:
            if ann['area'] > largest_area:
                largest_area = ann['area']
                largest_ann = ann

        if largest_ann is not None:
            mask = coco.annToMask(largest_ann)

        image, mask = preprocess(image, mask)

        mask = mask.squeeze()

        image = torch.tensor(image).permute(2,0,1)  # HWC → CHW
        # mask = torch.tensor(mask).permute(2,0,1)
        mask = torch.tensor(mask).unsqueeze(0)

        return image.float(), mask.float()