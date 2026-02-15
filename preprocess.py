import cv2
import numpy as np

def preprocess_image_and_mask(image, mask, size=(256, 256)):
    """
    Preprocess image and mask:
    1. Resize
    2. Normalize image
    3. Ensure mask remains binary
    """

    # Resize
    image_resized = cv2.resize(image, size)
    mask_resized = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    # Normalize image (0-255 -> 0-1)
    image_normalized = image_resized / 255.0

    return image_resized, image_normalized, mask_resized


def apply_mask(image, mask):
    """
    Apply binary mask to image to create isolated output
    """

    # Convert mask to 3-channel
    mask_3channel = np.stack([mask]*3, axis=-1)

    # Apply mask
    isolated = image * mask_3channel

    return isolated
