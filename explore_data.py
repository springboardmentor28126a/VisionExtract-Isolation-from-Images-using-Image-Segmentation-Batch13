import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import matplotlib.gridspec as gridspec

def explore_dataset(root_dir, subset='val2017', target_category='person'):
    """
    Loads a random image from the COCO dataset and visualizes the binary mask
    for a specific category (e.g., 'person').
    
    Args:
        root_dir (str): Root directory of the data (e.g., 'data/raw').
        subset (str): The subset to load (e.g., 'val2017').
        target_category (str): The COCO category to isolate (subject).
    """
    
    # Paths
    ann_file = os.path.join(root_dir, 'annotations', f'instances_{subset}.json')
    img_dir = os.path.join(root_dir, subset)

    # Initialize COCO api
    print(f"Loading annotations from {ann_file}...")
    coco = COCO(ann_file)
    
    # Get all image IDs containing the target category
    cat_ids = coco.getCatIds(catNms=[target_category])
    img_ids = coco.getImgIds(catIds=cat_ids)
    
    if not img_ids:
        print(f"No images found for category: {target_category}")
        return

    # Pick a random image
    random_img_id = random.choice(img_ids)
    img_info = coco.loadImgs(random_img_id)[0]
    img_path = os.path.join(img_dir, img_info['file_name'])

    # 1. Load Image
    print(f"Loading image: {img_info['file_name']}")
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for matplotlib

    # 2. Generate Binary Mask
    # Get annotations for this image and category
    ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    # Create an empty mask (black)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Combine all instances of 'person' into one binary mask
    for ann in anns:
        # coco.annToMask generates a binary mask for a single annotation
        pixel_mask = coco.annToMask(ann)
        # Combine with existing mask (Logical OR)
        mask = np.maximum(mask, pixel_mask)

    # 3. Visualize
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

    # Original Image
    ax0 = plt.subplot(gs[0])
    ax0.imshow(image)
    ax0.set_title(f"Original: {img_info['file_name']}")
    ax0.axis('off')

    # Binary Mask (Ground Truth)
    ax1 = plt.subplot(gs[1])
    ax1.imshow(mask, cmap='gray')
    ax1.set_title("Generated Binary Mask")
    ax1.axis('off')

    # Overlay
    ax2 = plt.subplot(gs[2])
    ax2.imshow(image)
    ax2.imshow(mask, alpha=0.5, cmap='jet') # Overlay mask with transparency
    ax2.set_title("Overlay")
    ax2.axis('off')

    
    plt.show()
    print("Check this image to verify that the mask perfectly covers the subject.")

if __name__ == "__main__":
    # Ensure you point to where 'download_data.py' saved the files
    explore_dataset(root_dir='data/raw')
