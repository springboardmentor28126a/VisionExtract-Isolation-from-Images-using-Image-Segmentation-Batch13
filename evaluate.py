"""
VisionExtract — Evaluation Script
===================================
Runs the full validation set through the best checkpoint and prints
a detailed report of all segmentation metrics.

Usage:
    python evaluate.py                                 # use default best checkpoint
    python evaluate.py --ckpt outputs/checkpoints/last_model.pth
    python evaluate.py --max_val 1000                  # quick eval
    python evaluate.py --save_grids                    # save comparison images
"""
import argparse
import os
import json

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import config
from data.coco_dataset import COCOSegmentationDataset
from data.transforms import get_val_transforms
from models.segmentation_model import load_model
from utils.metrics import compute_all_metrics, MetricTracker
from utils.visualize import save_comparison_grid, tensor_to_numpy_image, apply_mask_to_image


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VisionExtract model")
    p.add_argument("--ckpt",       type=str, default=config.BEST_CKPT)
    p.add_argument("--max_val",    type=int, default=None)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="Binarisation threshold for predictions")
    p.add_argument("--save_grids", action="store_true",
                   help="Save before/after comparison grids")
    p.add_argument("--n_grids",    type=int, default=16,
                   help="Number of sample grids to save")
    return p.parse_args()


def plot_metrics_bar(metrics: dict, save_path: str):
    """Save a bar chart of all evaluation metrics."""
    labels = list(metrics.keys())
    values = [metrics[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452", "#55A868",
                                          "#C44E52", "#8172B2"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("VisionExtract — Evaluation Metrics")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Eval] Metrics chart saved: {save_path}")


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = (torch.device("cuda")  if torch.cuda.is_available() else
              torch.device("mps")   if torch.backends.mps.is_available() else
              torch.device("cpu"))
    print(f"[Device] {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(args.ckpt, device)

    # ── Data ──────────────────────────────────────────────────────────────────
    val_ds = COCOSegmentationDataset(
        img_dir     = config.VAL_IMG_DIR,
        ann_file    = config.VAL_ANN_FILE,
        transforms  = get_val_transforms(),
        max_samples = args.max_val,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = True,
    )
    print(f"[Eval] Evaluating on {len(val_ds)} images …")

    # ── Per-image metrics ─────────────────────────────────────────────────────
    tracker   = MetricTracker()
    per_image = []          # list of per-image metric dicts

    saved_grids = 0
    grid_dir    = os.path.join(config.PRED_DIR, "eval_grids")
    if args.save_grids:
        os.makedirs(grid_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            preds  = torch.sigmoid(logits).cpu().numpy()   # (B,1,H,W)
            gts    = masks.cpu().numpy()                   # (B,1,H,W)

            for i in range(images.size(0)):
                m = compute_all_metrics(preds[i, 0], gts[i, 0], args.threshold)
                tracker.update(m)
                per_image.append(m)

                # Optionally save comparison grids
                if args.save_grids and saved_grids < args.n_grids:
                    orig  = tensor_to_numpy_image(images[i].cpu())
                    iso   = apply_mask_to_image(orig, preds[i, 0], args.threshold)
                    save_comparison_grid(
                        original  = orig,
                        gt_mask   = gts[i, 0],
                        pred_mask = preds[i, 0],
                        isolated  = iso,
                        save_path = os.path.join(grid_dir, f"eval_{saved_grids:04d}.png"),
                        title     = f"IoU={m['iou']:.3f} | Dice={m['dice']:.3f}",
                    )
                    saved_grids += 1

    # ── Aggregate results ─────────────────────────────────────────────────────
    avg = tracker.averages()

    print("\n" + "=" * 55)
    print("  VisionExtract — Evaluation Report")
    print("=" * 55)
    print(f"  Checkpoint  : {os.path.basename(args.ckpt)}")
    print(f"  Dataset size: {len(val_ds)} images")
    print(f"  Threshold   : {args.threshold}")
    print("-" * 55)
    print(f"  IoU (Jaccard)     : {avg['iou']:.4f}")
    print(f"  Dice Coefficient  : {avg['dice']:.4f}")
    print(f"  Pixel Accuracy    : {avg['pixel_acc']:.4f}")
    print(f"  Precision         : {avg['precision']:.4f}")
    print(f"  Recall            : {avg['recall']:.4f}")
    print("=" * 55)

    # Histogram of per-image IoU
    ious = [m["iou"] for m in per_image]
    print(f"\n  Per-image IoU distribution:")
    print(f"    Min  : {min(ious):.4f}")
    print(f"    Max  : {max(ious):.4f}")
    print(f"    Mean : {np.mean(ious):.4f}")
    print(f"    Std  : {np.std(ious):.4f}")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(ious, bins=bins)
    print("\n  IoU histogram:")
    for lo, hi, cnt in zip(bins[:-1], bins[1:], hist):
        bar = "#" * (cnt * 40 // max(hist + [1]))
        print(f"    [{lo:.1f}-{hi:.1f}] {cnt:5d} {bar}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(config.PRED_DIR, exist_ok=True)

    report = {"summary": avg, "per_image": per_image,
              "checkpoint": args.ckpt, "threshold": args.threshold}
    report_path = os.path.join(config.PRED_DIR, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Eval] Full report saved: {report_path}")

    plot_metrics_bar(avg, os.path.join(config.PRED_DIR, "eval_metrics_bar.png"))


if __name__ == "__main__":
    main()
