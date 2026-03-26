"""
Visualization helpers: overlay masks, side-by-side grids, isolated subject.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def apply_mask_to_image(image: np.ndarray,
                        mask: np.ndarray,
                        threshold: float = 0.5) -> np.ndarray:
    """
    Returns a new image where background pixels are set to black.

    Parameters
    ----------
    image     : (H, W, 3) uint8 RGB image
    mask      : (H, W) float mask (0-1 range, e.g. sigmoid output)
    threshold : binarisation threshold

    Returns
    -------
    (H, W, 3) uint8 subject-isolated image
    """
    binary = (mask >= threshold).astype(np.uint8)   # (H, W)
    binary = binary[:, :, np.newaxis]               # (H, W, 1)
    return image * binary                            # broadcast over channels


def overlay_mask(image: np.ndarray,
                 mask: np.ndarray,
                 alpha: float = 0.4,
                 color=(0, 255, 0),
                 threshold: float = 0.5) -> np.ndarray:
    """
    Overlay a semi-transparent coloured mask on the original image.
    Useful for qualitative inspection during training.
    """
    binary  = (mask >= threshold).astype(np.uint8)
    overlay = image.copy()
    overlay[binary == 1] = (
        alpha * np.array(color) +
        (1 - alpha) * overlay[binary == 1]
    ).astype(np.uint8)
    return overlay


def save_comparison_grid(original: np.ndarray,
                         gt_mask: np.ndarray,
                         pred_mask: np.ndarray,
                         isolated: np.ndarray,
                         save_path: str,
                         title: str = "") -> None:
    """
    4-panel figure: Original | GT Mask | Pred Mask | Isolated Subject.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground-Truth Mask")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Predicted Mask")

    axes[3].imshow(isolated)
    axes[3].set_title("Isolated Subject")

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def tensor_to_numpy_image(tensor) -> np.ndarray:
    """Convert a normalised (3, H, W) torch tensor → (H, W, 3) uint8 for display."""
    import torch
    import config
    mean = np.array(config.MEAN)
    std  = np.array(config.STD)

    img = tensor.cpu().numpy().transpose(1, 2, 0)   # (H, W, 3) float
    img = img * std + mean                           # de-normalise
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img
