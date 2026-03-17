import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import our custom modules
from src.dataset import CocoSegmentationDataset, get_validation_augmentation
from src.model import VisionExtractModel

def calculate_metrics(pred_mask, true_mask):
    """
    Calculates Intersection over Union (IoU) and Dice Score for a single batch.
    
    Args:
        pred_mask (torch.Tensor): The thresholded binary predictions.
        true_mask (torch.Tensor): The ground truth binary masks.
        
    Returns:
        tuple: (iou, dice) scores as floats.
    """
    pred = pred_mask.view(-1).float()
    truth = true_mask.view(-1).float()
    
    intersection = (pred * truth).sum().item()
    total_pixels = pred.sum().item() + truth.sum().item()
    union = total_pixels - intersection
    
    iou = intersection / (union + 1e-6)
    dice = (2. * intersection) / (total_pixels + 1e-6)
    
    return iou, dice

def evaluate_model(weights_path, arch, device):
    """
    Loads a specific model architecture and weights, then runs evaluation 
    against the local validation dataset.
    """
    print(f"--- Starting Evaluation ---")
    print(f"Architecture: {arch.upper()}")
    print(f"Weights: {weights_path}")
    print(f"Device: {device}")

    # 1. Load Model
    model = VisionExtractModel(arch=arch, encoder_name="resnet34").to(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Successfully loaded weights.")
    except FileNotFoundError:
        print(f"Error: Could not find weights at {weights_path}")
        return

    model.eval()

    # 2. Load Validation Dataset
    # We use the local val2017 dataset for consistent benchmarking
    test_dataset = CocoSegmentationDataset(
        root_dir='data/raw', 
        subset='val2017', 
        transform=get_validation_augmentation()
    )
    # Batch size 4 is safe for local CPU evaluation, increase if using GPU
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 3. Evaluation Loop
    total_iou = 0.0
    total_dice = 0.0
    num_batches = len(test_loader)

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating Test Set"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            # Convert raw logits to probabilities, then threshold at 0.5 for binary mask
            preds = torch.sigmoid(outputs) > 0.5 
            
            iou, dice = calculate_metrics(preds, masks)
            total_iou += iou
            total_dice += dice

    mean_iou = total_iou / num_batches
    mean_dice = total_dice / num_batches

    print(f"\n--- Final Results ---")
    print(f"Mean IoU:         {mean_iou:.4f}")
    print(f"Mean Dice Score:  {mean_dice:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate VisionExtract Models")
    parser.add_argument("--weights", type=str, required=True, help="Path to the .pth weights file")
    parser.add_argument("--arch", type=str, default="unet", choices=["unet", "fpn", "deeplabv3plus"], help="Model architecture used")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_model(args.weights, args.arch, device)

if __name__ == "__main__":
    main()
