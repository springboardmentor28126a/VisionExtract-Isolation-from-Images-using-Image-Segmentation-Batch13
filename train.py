"""
VisionExtract — Training Script
=================================
Usage:
    python train.py                        # full COCO training
    python train.py --max_train 5000       # quick experiment with 5k images
    python train.py --arch DeepLabV3Plus   # different architecture
    python train.py --resume outputs/checkpoints/last_model.pth
"""
import argparse
import os
import sys
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

import config
from data.coco_dataset import build_datasets
from models.segmentation_model import build_model
from utils.losses import CombinedLoss
from utils.metrics import compute_all_metrics, MetricTracker
from utils.visualize import save_comparison_grid, tensor_to_numpy_image, apply_mask_to_image


# ─── CLI args ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train VisionExtract segmentation model")
    p.add_argument("--arch",       type=str, default=config.ARCHITECTURE,
                   help="Model architecture")
    p.add_argument("--encoder",    type=str, default=config.ENCODER)
    p.add_argument("--epochs",     type=int, default=config.EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=config.LR)
    p.add_argument("--max_train",  type=int, default=None,
                   help="Cap training set size (quick experiments)")
    p.add_argument("--max_val",    type=int, default=2000,
                   help="Cap validation set size")
    p.add_argument("--resume",     type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no_aug",     action="store_true",
                   help="Disable training augmentation (for debugging)")
    return p.parse_args()


# ─── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("[Device] Apple MPS")
    else:
        d = torch.device("cpu")
        print("[Device] CPU (training will be slow!)")
    return d


# ─── One epoch ─────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device,
              is_train: bool, tracker: MetricTracker):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    tracker.reset()

    ctx = torch.enable_grad if is_train else torch.no_grad
    with ctx():
        for images, masks in tqdm(loader, leave=False,
                                  desc="train" if is_train else "val"):
            images = images.to(device, non_blocking=True)  # (B,3,H,W)
            masks  = masks.to(device, non_blocking=True)   # (B,1,H,W)

            logits = model(images)                         # (B,1,H,W)
            loss   = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # Per-batch metrics
            with torch.no_grad():
                preds = torch.sigmoid(logits).cpu().numpy()
                gts   = masks.cpu().numpy()
                for p, g in zip(preds, gts):
                    m = compute_all_metrics(p[0], g[0])
                    tracker.update(m)

    avg_loss = total_loss / len(loader)
    avg_metrics = tracker.averages()
    return avg_loss, avg_metrics


# ─── Save sample predictions ───────────────────────────────────────────────────
def save_val_samples(model, val_loader, device, epoch: int, n: int = 4):
    model.eval()
    out_dir = os.path.join(config.PRED_DIR, f"epoch_{epoch:03d}")
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for images, masks in val_loader:
            if saved >= n:
                break
            images = images.to(device)
            logits = model(images)
            preds  = torch.sigmoid(logits).cpu().numpy()   # (B,1,H,W)

            for i in range(min(images.size(0), n - saved)):
                orig_np = tensor_to_numpy_image(images[i].cpu())
                gt_np   = masks[i, 0].numpy()               # (H,W)
                pred_np = preds[i, 0]                       # (H,W)
                iso     = apply_mask_to_image(orig_np, pred_np)

                save_comparison_grid(
                    original   = orig_np,
                    gt_mask    = gt_np,
                    pred_mask  = pred_np,
                    isolated   = iso,
                    save_path  = os.path.join(out_dir, f"sample_{saved:02d}.png"),
                    title      = f"Epoch {epoch} | Sample {saved}",
                )
                saved += 1


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = get_device()

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.PRED_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR,  exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[Data] Building datasets …")
    train_ds, val_ds = build_datasets(
        max_train = args.max_train,
        max_val   = args.max_val,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = config.NUM_WORKERS,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[Model] Building {args.arch} + {args.encoder} …")
    model = build_model(architecture=args.arch, encoder=args.encoder).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {total_params:,}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY
    )
    if config.SCHEDULER == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    criterion = CombinedLoss()

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch  = 0
    best_iou     = 0.0
    history      = []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_iou    = ckpt.get("best_iou", 0.0)
        print(f"[Resume] Resumed from epoch {ckpt['epoch']}, best IoU={best_iou:.4f}")

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    train_tracker = MetricTracker()
    val_tracker   = MetricTracker()

    print(f"\n[Train] Starting training for {args.epochs} epochs …\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        train_loss, train_m = run_epoch(
            model, train_loader, criterion, optimizer, device,
            is_train=True, tracker=train_tracker,
        )

        # ── Validate ──────────────────────────────────────────────────────────
        val_loss, val_m = run_epoch(
            model, val_loader, criterion, optimizer, device,
            is_train=False, tracker=val_tracker,
        )

        # ── Scheduler step ────────────────────────────────────────────────────
        if config.SCHEDULER == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        elapsed = time.time() - t0

        # ── Logging ───────────────────────────────────────────────────────────
        print(
            f"Epoch [{epoch+1:03d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_m['iou']:.4f} | "
            f"Val Dice: {val_m['dice']:.4f} | "
            f"Val PixAcc: {val_m['pixel_acc']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        for k, v in val_m.items():
            writer.add_scalar(f"Metrics/val_{k}", v, epoch)
        for k, v in train_m.items():
            writer.add_scalar(f"Metrics/train_{k}", v, epoch)

        row = {"epoch": epoch+1, "train_loss": train_loss,
               "val_loss": val_loss, **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)

        # ── Save checkpoints ──────────────────────────────────────────────────
        ckpt = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_iou":         val_m["iou"],
            "best_iou":        best_iou,
            "config": {
                "arch":    args.arch,
                "encoder": args.encoder,
            },
        }

        # Always save last
        torch.save(ckpt, os.path.join(config.CKPT_DIR, "last_model.pth"))

        # Save best
        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            ckpt["best_iou"] = best_iou
            torch.save(ckpt, config.BEST_CKPT)
            print(f"  ** New best IoU: {best_iou:.4f} — checkpoint saved **")

        # Save prediction grids every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_val_samples(model, val_loader, device, epoch + 1)

    # ── Final ─────────────────────────────────────────────────────────────────
    writer.close()

    history_path = os.path.join(config.LOG_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Done] Best Val IoU: {best_iou:.4f}")
    print(f"[Done] History saved to: {history_path}")
    print(f"[Done] Best checkpoint: {config.BEST_CKPT}")


if __name__ == "__main__":
    main()
