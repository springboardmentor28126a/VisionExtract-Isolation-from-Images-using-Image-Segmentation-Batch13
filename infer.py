import argparse
import os

import cv2
import numpy as np
import torch

from model import UNet 
from preprocess import IMG_SIZE, imagenet_normalize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.4

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_model(weights_path: str = "best_unet.pth") -> torch.nn.Module:
    model = UNet().to(DEVICE)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_single_image(image_path: str) -> torch.Tensor:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = imagenet_normalize(img)

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor


def fill_holes(binary_mask_u8: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask using flood-fill from the border."""
    # binary_mask_u8 expected in {0, 255}
    h, w = binary_mask_u8.shape[:2]
    flood = binary_mask_u8.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Flood-fill background starting from (0,0) (top-left corner)
    cv2.floodFill(flood, mask, (0, 0), 255)

    # Holes are background regions not reached by flood fill.
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(binary_mask_u8, flood_inv)
    return filled


def keep_center_seed_component(
    hard_mask: np.ndarray,
    seed_ratio: float = 0.08,
    min_area_ratio: float = 0.005,
) -> np.ndarray:
    """
    Keep the connected component that intersects a small seed region near the image center.
    This is more robust than "single center pixel" when the center pixel is noisy.
    """
    if hard_mask.max() == 0:
        return hard_mask.astype(np.uint8)

    # Normalize mask to {0,255} for OpenCV connected-components.
    hard_u8 = (hard_mask > 0).astype(np.uint8) * 255

    h, w = hard_u8.shape[:2]
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * seed_ratio)
    r = max(2, r)

    seed = np.zeros_like(hard_u8, dtype=np.uint8)
    cv2.circle(seed, (cx, cy), r, 255, thickness=-1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hard_u8, connectivity=8)
    if num_labels <= 1:
        return (hard_u8 > 0).astype(np.uint8)

    # Candidate labels are those that intersect the seed.
    seed_labels = np.unique(labels[(seed > 0)])
    seed_labels = seed_labels[seed_labels != 0]

    # Filter by minimum area to remove small specks.
    min_area = int(h * w * min_area_ratio)

    best_label = None
    best_area = -1
    for lbl in seed_labels:
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_label = int(lbl)
            best_area = area

    if best_label is None:
        # Fallback: pick the largest component that meets min_area.
        best_label = None
        best_area = -1
        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area >= min_area and area > best_area:
                best_label = lbl
                best_area = area

    if best_label is None:
        # Absolute fallback: largest component.
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = 1 + int(np.argmax(areas))

    return (labels == best_label).astype(np.uint8)


def postprocess_mask(
    logits: torch.Tensor,
    threshold: float = THRESHOLD,
    seed_ratio: float = 0.08,
    min_area_ratio: float = 0.005,
) -> np.ndarray:
  
    probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()  

    # Light smoothing reduces speckle before thresholding.
    probs = cv2.GaussianBlur(probs, (3, 3), 0)

    # Use a hard mask only to find the main component, then apply that as a gate.
    hard = (probs > threshold).astype(np.uint8)

    # Cleanup: close small gaps, then open speckle/noise.
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    hard = keep_center_seed_component(hard, seed_ratio=seed_ratio, min_area_ratio=min_area_ratio)

    # Fill holes so parts like hair/arms don't break into background holes.
    hard_u8 = (hard > 0).astype(np.uint8) * 255
    hard_filled = fill_holes(hard_u8)
    hard = (hard_filled > 0).astype(np.uint8)

    # Zero out everything outside the main subject but keep soft edges inside
    alpha = probs * hard
    return alpha


def de_normalize(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return img


def run_inference(image_path: str, output_path: str, model: torch.nn.Module) -> None:
    inp = preprocess_single_image(image_path)

    with torch.no_grad():
        logits = model(inp)

    # Soft alpha mask in [0,1]
    alpha = postprocess_mask(logits) 
    img_vis = de_normalize(inp)      

    rgb = img_vis * alpha[:, :, None]
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    alpha_uint8 = (alpha * 255).astype(np.uint8)

    # Save as PNG with alpha channel so edges (hair, etc.) can be semi-transparent
    rgba = np.dstack([rgb_uint8, alpha_uint8])
    extracted_bgrA = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Ensure PNG extension if user gave e.g. .jpg
    if not output_path.lower().endswith(".png"):
        output_path = output_path + ".png"
    cv2.imwrite(output_path, extracted_bgrA)
    print(f"Saved isolated subject (RGBA PNG) to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Subject isolation inference")
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input image or folder"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to output image or folder"
    )

    args = parser.parse_args()

    model = load_model()

    if os.path.isdir(args.input):
        # Batch mode: process all images in folder
        os.makedirs(args.output, exist_ok=True)
        for fname in os.listdir(args.input):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                in_path = os.path.join(args.input, fname)
                out_path = os.path.join(args.output, fname)
                run_inference(in_path, out_path, model)
    else:
        # Single image
        run_inference(args.input, args.output, model)


if __name__ == "__main__":
    main()

