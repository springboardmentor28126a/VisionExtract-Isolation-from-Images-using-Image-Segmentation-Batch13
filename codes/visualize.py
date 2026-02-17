import matplotlib.pyplot as plt
import cv2
import os
from generate_mask import generate_full_mask, IMAGE_DIR

def visualize_sample(image_id):
    mask, filename = generate_full_mask(image_id)

    image_path = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.title("Binary Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_sample(36678)