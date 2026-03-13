import cv2
import numpy as np

def clean_binary_mask(mask_numpy, kernel_size=5):
    """
    Applies morphological operations to clean up a binary mask.
    Removes small noise (Opening) and fills small holes (Closing).
    
    Args:
        mask_numpy (np.ndarray): The 2D binary mask (0s and 1s) from the model.
        kernel_size (int): Size of the morphological kernel.
        
    Returns:
        np.ndarray: The cleaned binary mask.
    """
    # Ensure mask is uint8 for OpenCV
    mask_uint8 = (mask_numpy * 255).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 1. Closing: Fills small holes inside the foreground object
    closing = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    # 2. Opening: Removes small noise/artifacts in the background
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
    # Convert back to 0/1 binary format
    clean_mask = (opening / 255.0).astype(np.float32)
    return clean_mask
