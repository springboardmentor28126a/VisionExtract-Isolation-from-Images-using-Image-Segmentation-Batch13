import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import segmentation_models_pytorch as smp
import albumentations as A

# ===== PATHS =====
COCO_ROOT = r"C:\Users\Thanuja\Desktop\vision_extraction\COCO Dataset\coco_2017"
IMAGE_DIR = os.path.join(COCO_ROOT, "train2017")
ANNOTATION_FILE = os.path.join(COCO_ROOT, "annotations", "instances_train2017.json")

# ===== LOAD COCO =====
coco = COCO(ANNOTATION_FILE)
image_id = coco.getImgIds()[0]  # use first image only

# ===== TRANSFORM =====
transform = A.Resize(256, 256)

# ===== DEVICE =====
device = "cpu"

# ===== LOAD MODEL =====
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)

model.load_state_dict(torch.load("week5_coco_model.pth", map_location=device))
model.to(device)
model.eval()

# ===== LOAD IMAGE =====
img_info = coco.loadImgs(image_id)[0]
img_path = os.path.join(IMAGE_DIR, img_info['file_name'])

image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ===== CREATE GROUND TRUTH MASK =====
mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
ann_ids = coco.getAnnIds(imgIds=image_id)
anns = coco.loadAnns(ann_ids)
for ann in anns:
    mask = np.maximum(mask, coco.annToMask(ann))

# ===== APPLY TRANSFORM =====
augmented = transform(image=image, mask=mask)
image = augmented["image"]
mask = augmented["mask"]

# ===== PREPARE INPUT =====
image_tensor = torch.tensor(np.transpose(image / 255.0, (2, 0, 1))).float().unsqueeze(0)

# ===== PREDICTION =====
with torch.no_grad():
    pred = model(image_tensor)
    pred = torch.sigmoid(pred).cpu().numpy()[0, 0]

pred_mask = (pred > 0.5).astype(np.uint8)

# ===== FINAL OUTPUT =====
final = image * np.expand_dims(pred_mask, axis=2)

# ===== DISPLAY =====
plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.imshow(image)
plt.title("Input")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(mask, cmap='gray')
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(pred_mask, cmap='gray')
plt.title("Prediction")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(final)
plt.title("Final Output")
plt.axis("off")

plt.tight_layout()
plt.show()