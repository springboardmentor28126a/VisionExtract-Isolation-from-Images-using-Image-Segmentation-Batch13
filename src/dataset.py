import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CocoSegmentationDataset(Dataset):
    """
    A PyTorch Dataset for Universal binary segmentation (Subject vs Background) using COCO.
    V2.0: Merges multiple target categories into a single 'subject' mask.
    
    Attributes:
        root_dir (str): Path to the dataset root (e.g., 'data/raw').
        subset (str): 'train2017' or 'val2017'.
        transform (albumentations.Compose): Transformations to apply to image and mask.
    """
    def __init__(self, root_dir, subset='val2017', transform=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        
        # Paths
        self.img_dir = os.path.join(root_dir, subset)
        self.ann_file = os.path.join(root_dir, 'annotations', f'instances_{subset}.json')
        
        # Initialize COCO API
        self.coco = COCO(self.ann_file)
        
        # V2.0 Universal Subject Definition
        target_categories = ['person', 'dog', 'cat', 'car', 'bicycle']
        self.cat_ids = self.coco.getCatIds(catNms=target_categories)
        
        # Filter images that contain AT LEAST ONE of the target categories
        all_img_ids = []
        for cat_id in self.cat_ids:
            all_img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))
            
        # Remove duplicate image IDs (e.g., an image with both a person and a dog)
        self.img_ids = list(set(all_img_ids))
        
        print(f"Dataset initialized. Found {len(self.img_ids)} images with target subjects in {subset}.")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Fetches a single image/mask pair, applies transforms, and returns tensors.
        """
        # 1. Get Image Info
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # 2. Load Image (OpenCV loads as BGR, convert to RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. Generate Universal Binary Mask
        # Load annotations for ALL target categories in this specific image
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        
        # Create a blank mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Merge all instance masks into one binary mask
        for ann in anns:
            pixel_mask = self.coco.annToMask(ann)
            mask = np.maximum(mask, pixel_mask) # np.maximum ensures overlap stays as 1
        
        # 4. Apply Augmentations (Albumentations handles both image and mask)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # 5. Final Formatting
        # Mask needs to be float32 for BCELoss later, and have a channel dimension (1, H, W)
        if isinstance(mask, torch.Tensor):
             mask = mask.float().unsqueeze(0) # Add channel dim -> (1, H, W)
        
        return image, mask

def get_training_augmentation():
    """
    V2.0 High-Resolution Training Pipeline (512x512)
    """
    train_transform = [
        A.Resize(height=512, width=512),
        A.HorizontalFlip(p=0.5),
        # Using Affine instead of deprecated ShiftScaleRotate
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.1), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    """
    V2.0 High-Resolution Validation Pipeline (512x512)
    No flipping/randomness. Just Resize, Normalize, and Tensor conversion.
    """
    test_transform = [
        A.Resize(height=512, width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(test_transform)
