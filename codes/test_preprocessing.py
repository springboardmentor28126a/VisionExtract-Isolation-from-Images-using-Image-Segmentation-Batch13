import os
import cv2
import matplotlib.pyplot as plt
from generate_mask import generate_full_mask, IMAGE_DIR
from preprocessing import preprocess

def test_sample(image_id):
    mask, filename = generate_full_mask(image_id)

    image_path = os.path.join(IMAGE_DIR, filename)
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Preprocess
    processed_image, processed_mask = preprocess(original_image.copy(), mask.copy(), augment=True)

    plt.figure(figsize=(15,5))

    # Original Image
    plt.subplot(1,3,1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    # Preprocessed Image
    plt.subplot(1,3,2)
    plt.imshow(processed_image)
    plt.title("Preprocessed Image (Resized + Normalized)")
    plt.axis("off")

    # Binary Mask
    plt.subplot(1,3,3)
    plt.imshow(processed_mask.squeeze(), cmap='gray')
    plt.title("Binary Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_sample(572303)