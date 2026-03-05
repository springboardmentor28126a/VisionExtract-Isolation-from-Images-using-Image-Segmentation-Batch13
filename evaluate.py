import torch
from torch.utils.data import DataLoader, random_split
from torch_dataset import SegmentationTorchDataset
from model import UNet
from metrics import dice_score, iou_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
dataset = SegmentationTorchDataset(
    image_dir="images/train",
    mask_dir="masks/train"
)

# Same split as training
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

_, _, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load model
model = UNet().to(DEVICE)
model.load_state_dict(torch.load("unet_week4.pth", map_location=DEVICE))
model.eval()

total_dice = 0
total_iou = 0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)

        total_dice += dice_score(preds, masks).item()
        total_iou += iou_score(preds, masks).item()

avg_dice = total_dice / len(test_loader)
avg_iou = total_iou / len(test_loader)

print(f"Average Dice Score: {avg_dice:.4f}")
print(f"Average IoU Score: {avg_iou:.4f}")