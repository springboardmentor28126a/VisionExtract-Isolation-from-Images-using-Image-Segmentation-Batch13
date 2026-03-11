from pycocotools.coco import COCO
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Load COCO annotations
coco = COCO("C:/Users/Thanuja/Downloads/COCO Dataset/coco2017/annotations/instances_train2017.json")

img_ids = coco.getImgIds()

for img_id in img_ids[:5]:

    img_info = coco.loadImgs(img_id)[0]

    image_path = os.path.join(
        "C:/Users/Thanuja/Downloads/COCO Dataset/coco2017/train2017",
        img_info['file_name']
    )

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get original size
    orig_h, orig_w, _ = image.shape

    # Create mask
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

    for ann in anns:
        mask = np.maximum(mask, coco.annToMask(ann))

    # Resize
    image_resized = cv2.resize(image, (256, 256))
    mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

    mask_resized = (mask_resized > 0).astype(np.uint8)

    # Get resized size
    res_h, res_w, _ = image_resized.shape

    # Normalize
    image_normalized = image_resized / 255.0

    # Isolate
    isolated = image_resized * mask_resized[:, :, np.newaxis]

    # Display
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title(f"Original\nSize: {orig_w}x{orig_h}")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title(f"Resized\nSize: {res_w}x{res_h}")
    plt.imshow(image_resized)
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Isolated")
    plt.imshow(isolated)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    input("press enter to continue...")

    print("Processed:", img_info['file_name'])
    print("Original Shape:", image.shape)
    print("Resized Shape:", image_resized.shape)