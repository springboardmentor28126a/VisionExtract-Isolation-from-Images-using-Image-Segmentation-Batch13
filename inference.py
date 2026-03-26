"""
VisionExtract — Inference Script
===================================
Run subject isolation on single images or a whole folder.

Usage:
    python inference.py --input photo.jpg
    python inference.py --input /path/to/folder --output /path/to/results
    python inference.py --input photo.jpg --show
"""
import argparse
import os
import sys
import glob

import cv2
import numpy as np
import torch
from PIL import Image

import config
from models.segmentation_model import load_model
from data.transforms import get_inference_transforms
from utils.visualize import apply_mask_to_image, overlay_mask


# ─── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="VisionExtract inference")
    p.add_argument("--input",     type=str, required=True,
                   help="Path to an image file OR a directory of images")
    p.add_argument("--output",    type=str, default="outputs/predictions/inference",
                   help="Directory to save results")
    p.add_argument("--ckpt",      type=str, default=config.BEST_CKPT,
                   help="Model checkpoint path")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Mask binarisation threshold (0-1)")
    p.add_argument("--show",      action="store_true",
                   help="Open a window with the result (requires a display)")
    p.add_argument("--overlay",   action="store_true",
                   help="Also save a semi-transparent overlay image")
    return p.parse_args()


# ─── Inference on a single image ───────────────────────────────────────────────
def predict_single(
    image_bgr: np.ndarray,
    model: torch.nn.Module,
    transforms,
    device: torch.device,
    threshold: float = 0.5,
):
    """
    Parameters
    ----------
    image_bgr : OpenCV-loaded image (H, W, 3) BGR uint8

    Returns
    -------
    isolated  : (H, W, 3) uint8 — subject only, background = black
    pred_mask : (H, W) float — raw sigmoid probability map
    binary    : (H, W) uint8 — binarised mask
    """
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── Preprocess ────────────────────────────────────────────────────────────
    aug     = transforms(image=image_rgb, mask=np.zeros((orig_h, orig_w), dtype=np.uint8))
    tensor  = aug["image"].unsqueeze(0).to(device)  # (1, 3, H, W)

    # ── Forward pass ──────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logit    = model(tensor)                         # (1, 1, H, W)
        pred_mask = torch.sigmoid(logit)[0, 0].cpu().numpy()  # (H, W)

    # ── Resize mask back to original resolution ───────────────────────────────
    pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    binary    = (pred_mask >= threshold).astype(np.uint8)

    # ── Apply mask ────────────────────────────────────────────────────────────
    isolated = apply_mask_to_image(image_rgb, pred_mask, threshold)

    return isolated, pred_mask, binary


# ─── Collect image paths ───────────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_paths(input_path: str):
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        paths = []
        for f in sorted(os.listdir(input_path)):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(input_path, f))
        return paths
    raise FileNotFoundError(f"Input not found: {input_path}")


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Device
    device = (torch.device("cuda")  if torch.cuda.is_available() else
              torch.device("mps")   if torch.backends.mps.is_available() else
              torch.device("cpu"))
    print(f"[Inference] Device: {device}")

    # Model
    model      = load_model(args.ckpt, device)
    transforms = get_inference_transforms()

    # Output dir
    os.makedirs(args.output, exist_ok=True)

    # Collect input files
    paths = collect_paths(args.input)
    print(f"[Inference] Processing {len(paths)} image(s) …")

    for img_path in paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]

        # Load
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"  [SKIP] Cannot read: {img_path}")
            continue

        # Predict
        isolated, pred_mask, binary = predict_single(
            bgr, model, transforms, device, args.threshold
        )

        # Save isolated subject (RGB → BGR for cv2)
        out_isolated = os.path.join(args.output, f"{stem}_isolated.png")
        cv2.imwrite(out_isolated, cv2.cvtColor(isolated, cv2.COLOR_RGB2BGR))

        # Save binary mask
        out_mask = os.path.join(args.output, f"{stem}_mask.png")
        cv2.imwrite(out_mask, binary * 255)

        # Optionally save overlay
        if args.overlay:
            orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ov       = overlay_mask(orig_rgb, pred_mask, alpha=0.4)
            out_ov   = os.path.join(args.output, f"{stem}_overlay.png")
            cv2.imwrite(out_ov, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

        # Show in window (if requested and display is available)
        if args.show:
            orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            combined = np.concatenate([orig_rgb, isolated], axis=1)
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            cv2.imshow("Original | Isolated", combined_bgr)
            cv2.waitKey(0)

        print(f"  Saved: {out_isolated}")

    if args.show:
        cv2.destroyAllWindows()

    print(f"\n[Inference] Done. Results in: {args.output}")


if __name__ == "__main__":
    main()
