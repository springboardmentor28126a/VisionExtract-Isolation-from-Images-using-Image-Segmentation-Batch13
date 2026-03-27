import torch
import time
import os
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast
from torch_dataset import SegmentationTorchDataset
from model import UNet
from metrics import dice_score, iou_score

assert torch.cuda.is_available(), "CUDA GPU is required but not available."
DEVICE = "cuda"

def main():
    t0 = time.perf_counter()
    print("Starting evaluation...", flush=True)
    torch.backends.cudnn.benchmark = True

    print("Loading dataset...", flush=True)
    dataset = SegmentationTorchDataset(
        image_dir="images/train",
        mask_dir="masks/train"
    )

    # Same split as training (seeded)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)

    _, _, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    worker_count = min(4, os.cpu_count() or 0)
    print(f"Building DataLoader (num_workers={worker_count})...", flush=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=True,
        persistent_workers=worker_count > 0,
    )

    print("Loading model...", flush=True)
    model = UNet().to(DEVICE)
    try:
        state_dict = torch.load("best_unet.pth", map_location="cuda", weights_only=True)
    except TypeError:
        # Older PyTorch: weights_only not supported
        state_dict = torch.load("best_unet.pth", map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()

    total_dice = 0.0
    total_iou = 0.0

    print(f"Running evaluation on {len(test_loader)} batches...", flush=True)
    eval_t0 = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader, start=1):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.float16):
                preds = model(images)

            total_dice += dice_score(preds, masks).item()
            total_iou += iou_score(preds, masks).item()

            if batch_idx % 20 == 0 or batch_idx == len(test_loader):
                print(f"Processed {batch_idx}/{len(test_loader)} batches...", flush=True)

    eval_dt = time.perf_counter() - eval_t0
    avg_dice = total_dice / len(test_loader)
    avg_iou = total_iou / len(test_loader)

    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    print(f"Evaluation time: {eval_dt:.2f}s")
    print(f"Total runtime: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
