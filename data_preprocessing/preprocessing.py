from pycocotools.coco import COCO
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


annotation_path = "C:/Users/varsh/Downloads/archive/coco2017/annotations/instances_val2017.json"
image_folder = "C:/Users/varsh/Downloads/archive/coco2017/val2017"
output_base = "C:/Users/varsh/Downloads/archive/coco2017/processed_val"


# Create output folders
images_out = os.path.join(output_base, "images")
masks_out = os.path.join(output_base, "masks")
isolated_out = os.path.join(output_base, "isolated")

os.makedirs(images_out, exist_ok=True)
os.makedirs(masks_out, exist_ok=True)
os.makedirs(isolated_out, exist_ok=True)

# Load COCO validation annotations
coco = COCO(annotation_path)

img_ids = coco.getImgIds()
print("Total validation images:", len(img_ids))

display_count = 0  # To display only first 5 images

for img_id in img_ids:

    img_info = coco.loadImgs(img_id)[0]
    image_path = os.path.join(image_folder, img_info['file_name'])

    image = cv2.imread(image_path)
    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create mask
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

    for ann in anns:
        mask = np.maximum(mask, coco.annToMask(ann))

    # Resize
    image_resized = cv2.resize(image, (256, 256))
    mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

    # Convert mask to binary
    mask_resized = (mask_resized > 0).astype(np.uint8)

    # Normalize
    image_normalized = image_resized / 255.0

    # Isolate subject
    isolated = image_resized * mask_resized[:, :, np.newaxis]

    # =========================
    # SAVE OUTPUTS
    # =========================

    cv2.imwrite(
        os.path.join(images_out, img_info['file_name']),
        cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        os.path.join(masks_out, img_info['file_name']),
        mask_resized * 255
    )

    cv2.imwrite(
        os.path.join(isolated_out, img_info['file_name']),
        cv2.cvtColor(isolated, cv2.COLOR_RGB2BGR)
    )

    print("Processed:", img_info['file_name'])

    # =========================
    # DISPLAY ONLY FIRST 5
    # =========================
    if display_count < 5:

        plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        plt.title("Original")
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.title("Mask")
        plt.imshow(mask_resized, cmap='gray')
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.title("Isolated")
        plt.imshow(isolated)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        display_count += 1

print("✅ All validation images processed, saved and sample displayed successfully!")