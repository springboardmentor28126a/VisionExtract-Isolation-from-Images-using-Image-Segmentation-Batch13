"""
VisionExtract — Gradio Web Application
========================================
Upload any image and receive:
  1. The subject-isolated result (background = black)
  2. The binary mask
  3. A side-by-side comparison

Usage:
    python app.py                       # uses best checkpoint by default
    python app.py --ckpt path/to/ckpt.pth
    python app.py --share               # create a public Gradio share link
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image

import config
from models.segmentation_model import load_model
#from data.transforms import get_inference_transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_inference_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])
from utils.visualize import apply_mask_to_image, overlay_mask


# ─── Global model (loaded once at startup) ─────────────────────────────────────
_model      = None
_transforms = None
_device     = None


def load_resources(ckpt_path: str):
    global _model, _transforms, _device

    _device = (torch.device("cuda")  if torch.cuda.is_available() else
               torch.device("mps")   if torch.backends.mps.is_available() else
               torch.device("cpu"))

    print(f"[App] Device: {_device}")

    if not os.path.isfile(ckpt_path):
        print(f"[App] WARNING: checkpoint not found at {ckpt_path}.")
        print("[App] The app will start, but inference will fail until a "
              "checkpoint is provided.")
        return

    _model      = load_model(ckpt_path, _device)
    _transforms = get_inference_transforms()
    print(f"[App] Model loaded from: {ckpt_path}")


# ─── Core inference function ────────────────────────────────────────────────────
def isolate_subject(
    pil_image: Image.Image,
    threshold: float = 0.5,
    show_overlay: bool = False,
) -> tuple:
    """
    Called by Gradio on every upload.

    Returns
    -------
    (isolated_pil, mask_pil, side_by_side_pil, stats_str)
    """
    if _model is None:
        err = "Model not loaded. Train the model first and provide a checkpoint."
        blank = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        return blank, blank, blank, err

    # PIL → numpy RGB
    image_rgb = np.array(pil_image.convert("RGB"))
    orig_h, orig_w = image_rgb.shape[:2]

    # Preprocess
    aug    = _transforms(image=image_rgb,
                         mask=np.zeros((orig_h, orig_w), dtype=np.uint8))
    tensor = aug["image"].unsqueeze(0).to(_device)

    # Forward
    _model.eval()
    with torch.no_grad():
        logit    = _model(tensor)
        pred_map = torch.sigmoid(logit)[0, 0].cpu().numpy()

    # Resize mask to original size
    pred_map = cv2.resize(pred_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    binary   = (pred_map >= threshold).astype(np.uint8)

    # Subject-isolated image
    isolated = apply_mask_to_image(image_rgb, pred_map, threshold)

    # Binary mask as greyscale (0-255)
    mask_vis = (binary * 255).astype(np.uint8)
    mask_rgb = np.stack([mask_vis] * 3, axis=-1)

    # Side-by-side: Original | Mask | Isolated
    divider = np.ones((orig_h, 4, 3), dtype=np.uint8) * 200  # grey divider
    if show_overlay:
        middle = overlay_mask(image_rgb, pred_map, alpha=0.5)
    else:
        middle = mask_rgb
    side_by_side = np.concatenate([image_rgb, divider, middle, divider, isolated],
                                  axis=1)

    # Stats string
    subject_pct = float(binary.mean()) * 100
    stats = (
        f"Subject coverage : {subject_pct:.1f}%\n"
        f"Background       : {100 - subject_pct:.1f}%\n"
        f"Image size       : {orig_w} x {orig_h}\n"
        f"Threshold        : {threshold:.2f}"
    )

    return (
        Image.fromarray(isolated),
        Image.fromarray(mask_rgb),
        Image.fromarray(side_by_side),
        stats,
    )


# ─── Gradio UI ─────────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="VisionExtract — AI Subject Isolation",
        theme=gr.themes.Soft(),
        css=".output-image { border-radius: 8px; }",
    ) as demo:

        gr.Markdown(
            """
            # VisionExtract
            ### AI-powered Subject Isolation from Images

            Upload any image and the model will automatically detect and extract the main subject,
            rendering all background pixels **black** — exactly like a photographer's cutout.

            > Powered by a U-Net + ResNet-50 encoder trained on COCO 2017.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil", label="Upload Image",
                    height=400,
                )
                threshold_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Mask Threshold",
                    info="Lower = more pixels included as subject",
                )
                overlay_toggle = gr.Checkbox(
                    label="Show mask overlay instead of binary mask",
                    value=False,
                )
                run_btn = gr.Button("Isolate Subject", variant="primary")

            with gr.Column(scale=2):
                with gr.Tab("Isolated Subject"):
                    isolated_out = gr.Image(
                        type="pil", label="Subject Only (background = black)",
                        height=400,
                    )
                with gr.Tab("Predicted Mask"):
                    mask_out = gr.Image(
                        type="pil", label="Binary Mask",
                        height=400,
                    )
                with gr.Tab("Side-by-Side"):
                    comparison_out = gr.Image(
                        type="pil", label="Original | Mask | Isolated",
                        height=400,
                    )
                stats_out = gr.Textbox(
                    label="Prediction Stats",
                    lines=4,
                    interactive=False,
                )

        # Examples (will work if sample images are present)
        sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
        if os.path.isdir(sample_dir):
            examples = [[os.path.join(sample_dir, f)] for f in os.listdir(sample_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))][:6]
            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=[input_image],
                    label="Example Images",
                )

        run_btn.click(
            fn=isolate_subject,
            inputs=[input_image, threshold_slider, overlay_toggle],
            outputs=[isolated_out, mask_out, comparison_out, stats_out],
        )

        # Also trigger on image upload
        input_image.upload(
            fn=isolate_subject,
            inputs=[input_image, threshold_slider, overlay_toggle],
            outputs=[isolated_out, mask_out, comparison_out, stats_out],
        )

        gr.Markdown(
            """
            ---
            **How it works:**
            1. The image is resized and normalised (ImageNet stats)
            2. A U-Net with ResNet-50 backbone predicts a per-pixel probability map
            3. Pixels above the threshold are kept; others become black
            4. The result is resized back to the original resolution

            **Metrics on COCO 2017 validation:**
            IoU · Dice Coefficient · Pixel Accuracy (see `evaluate.py` for full report)
            """
        )

    return demo


# ─── Entry point ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   type=str, default=config.BEST_CKPT)
    p.add_argument("--port",   type=int, default=7860)
    p.add_argument("--share",  action="store_true",
                   help="Generate a public Gradio share link")
    args = p.parse_args()

    load_resources(args.ckpt)

    demo = build_ui()
    demo.launch(
        server_port  = args.port,
        share        = args.share,
        show_error   = True,
    )


if __name__ == "__main__":
    main()
