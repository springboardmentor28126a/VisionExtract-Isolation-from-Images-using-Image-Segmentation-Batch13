import cv2
import numpy as np


def prepare_image(image, size=(256, 256)):
    """
    Resize and normalize image for model input
    """
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    return image


def apply_mask(image, mask):
    """
    Apply binary mask to extract subject
    """

    # Ensure mask is binary
    mask = (mask > 0.5).astype(np.uint8)

    # Convert to 3-channel
    mask = np.stack([mask] * 3, axis=-1)

    # Apply mask
    result = image * mask

    return result