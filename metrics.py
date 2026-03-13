import torch
from torch.utils.data import DataLoader, Subset
from dataset import COCODataset
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = COCODataset(
    image_dir="val2017",
    ann_file="annotations/instances_val2017.json"
)

# Use only 50 images for fast evaluation
dataset = Subset(dataset, range(50))

loader = DataLoader(dataset, batch_size=1, shuffle=False)

print("Dataset loaded")
print("Total samples:", len(loader))


model = get_model(num_classes=1)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()


def compute_iou(pred, mask):

    pred = pred.bool()
    mask = mask.bool()

    intersection = (pred & mask).float().sum()
    union = (pred | mask).float().sum()

    return (intersection + 1e-6) / (union + 1e-6)


def compute_dice(pred, mask):

    pred = pred.bool()
    mask = mask.bool()

    intersection = (pred & mask).float().sum()

    return (2 * intersection + 1e-6) / (pred.float().sum() + mask.float().sum() + 1e-6)


def compute_accuracy(pred, mask):

    pred = pred.bool()
    mask = mask.bool()

    correct = (pred == mask).float().sum()
    total = mask.numel()

    return correct / total


total_iou = 0
total_dice = 0
total_acc = 0
count = 0


print("\nEvaluating...\n")

with torch.no_grad():

    for i, (images, masks) in enumerate(loader):

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

        preds = preds.squeeze(1)
        masks = masks.squeeze(1)

        iou = compute_iou(preds, masks)
        dice = compute_dice(preds, masks)
        acc = compute_accuracy(preds, masks)

        total_iou += iou
        total_dice += dice
        total_acc += acc

        count += 1

        print(f"Processed image {i+1}/{len(loader)}")


print("\n----- Milestone 3 Metrics -----")
print("Mean IoU:", (total_iou/count).item())
print("Mean Dice Score:", (total_dice/count).item())
print("Pixel Accuracy:", (total_acc/count).item())