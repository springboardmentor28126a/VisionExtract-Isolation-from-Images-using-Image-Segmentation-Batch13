import torch
from torch.utils.data import DataLoader
from dataset import COCODataset
from model import get_model

def compute_iou(pred, mask):
    pred = pred.int()
    mask = mask.int()

    intersection = (pred & mask).float().sum((1,2))
    union = (pred | mask).float().sum((1,2))

    return (intersection / (union + 1e-6)).mean().item()

def compute_dice(pred, mask):
    pred = pred.float()
    mask = mask.float()

    intersection = (pred * mask).sum((1,2))
    return ((2 * intersection) / (pred.sum((1,2)) + mask.sum((1,2)) + 1e-6)).mean().item()

def compute_accuracy(pred, mask):
    return (pred == mask).float().mean().item()

device = torch.device("cpu")

dataset = COCODataset(
    image_dir="val2017",
    ann_file="annotations/instances_val2017.json"
)

loader = DataLoader(dataset, batch_size=2)

model = get_model()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

total_iou, total_dice, total_acc, count = 0,0,0,0

with torch.no_grad():
    for images, masks in loader:
        outputs = model(images)

        preds = (torch.sigmoid(outputs) > 0.5).int()

        total_iou += compute_iou(preds.squeeze(1), masks.squeeze(1))
        total_dice += compute_dice(preds.squeeze(1), masks.squeeze(1))
        total_acc += compute_accuracy(preds.squeeze(1), masks.squeeze(1))

        count += 1

print("IoU:", total_iou / count)
print("Dice Score:", total_dice / count)
print("Pixel Accuracy:", total_acc / count)