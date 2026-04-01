import os
import random
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import torch
import numpy as np

# -----------------------------
# Your COCODataset Class
# -----------------------------
class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_file, size=256, ids=None):
        self.image_dir = "C:\\Users\\Thanuja\\Desktop\\vision_extraction\\COCO Dataset\\coco_2017\\val2017"
        self.coco = COCO("C:\\Users\\Thanuja\\Desktop\\vision_extraction\\COCO Dataset\\coco_2017\\annotations\\instances_train2017.json")
        if ids is None:
            self.ids = list(self.coco.imgs.keys())
        else:
            self.ids = ids  # allow custom subset of IDs
        self.size = size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        image = cv2.imread(img_path)
        image = cv2.resize(image, (self.size, self.size))
        image = image / 255.0
        image = image.transpose(2, 0, 1)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.float32)
        for ann in anns:
            mask += self.coco.annToMask(ann)
        mask = (mask > 0).astype(np.float32)
        mask = cv2.resize(mask, (self.size, self.size))
        mask = mask[None, :, :]

        return torch.tensor(image).float(), torch.tensor(mask).float()


# -----------------------------
# Dataset Split Function
# -----------------------------
def split_dataset(image_dir, annotation_file, total_images=5000, seed=42):
    coco = COCO("C:\\Users\\Thanuja\\Desktop\\vision_extraction\\COCO Dataset\\coco_2017\\annotations\\instances_train2017.json")
    all_ids = list(coco.imgs.keys())[:total_images]  # take first 5000 images
    random.seed(seed)
    random.shuffle(all_ids)

    n_train = int(0.7 * total_images)    # 70% train
    n_val = int(0.15 * total_images)     # 15% validation
    n_test = total_images - n_train - n_val  # remaining 15% test

    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train:n_train + n_val]
    test_ids = all_ids[n_train + n_val:]

    train_dataset = COCODataset(image_dir, annotation_file, ids=train_ids)
    val_dataset = COCODataset(image_dir, annotation_file, ids=val_ids)
    test_dataset = COCODataset(image_dir, annotation_file, ids=test_ids)

    return train_dataset, val_dataset, test_dataset


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    image_dir = "C:\\Users\\Thanuja\\Desktop\\vision_extraction\\COCO Dataset\\coco_2017\\val2017"
    annotation_file = "C:\\Users\\Thanuja\\Desktop\\vision_extraction\\COCO Dataset\\coco_2017\\annotations\\instances_train2017.json"

    train_dataset, val_dataset, test_dataset = split_dataset(image_dir, annotation_file, total_images=5000)

    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    # Example DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)