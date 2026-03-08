import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split

# =========================================================
# Double Conv Block
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# =========================================================
# U-Net Model
# =========================================================
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final(d1))

# =========================================================
# Dice Loss
# =========================================================
class DiceLoss(nn.Module):
    def forward(self, preds, targets, smooth=1e-6):
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + smooth) / (
            preds.sum() + targets.sum() + smooth
        )
        return 1 - dice

# =========================================================
# Combined Loss 
# =========================================================
class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return self.bce(preds, targets) + self.dice(preds, targets)

# =========================================================
# Dataset
# =========================================================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.images = sorted(os.listdir(image_dir))  
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # ----- image -----
        img = cv2.imread(os.path.join(self.image_dir, img_name))
        img = cv2.resize(img, (128, 128)) / 255.0

        # ----- mask -----
        mask = cv2.imread(os.path.join(self.mask_dir, img_name), 0)
        mask = cv2.resize(mask, (128, 128))
        mask = (mask > 127).astype("float32")  

        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask

# =========================================================
# Metrics
# =========================================================
def iou_score(preds, targets):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def dice_score(preds, targets):
    preds = (preds > 0.5).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2 * intersection + 1e-6) / (
        preds.sum() + targets.sum() + 1e-6
    )

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_dir = "C:/Users/varsh/Downloads/archive/coco2017/processed_val/images"
    mask_dir = "C:/Users/varsh/Downloads/archive/coco2017/processed_val/masks"

    dataset = SegmentationDataset(image_dir, mask_dir)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    #  MODEL SETUP
    model = UNet().to(device)
    criterion = ComboLoss()              
    optimizer = optim.Adam(model.parameters(), lr=0.001)  
    num_epochs = 5                     

    # ================= TRAIN =================
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================= VALIDATION =================
        model.eval()
        val_iou, val_dice = 0, 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                val_iou += iou_score(outputs, masks).item()
                val_dice += dice_score(outputs, masks).item()

        val_iou /= len(val_loader)
        val_dice /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val IoU: {val_iou:.4f} | "
              f"Val Dice: {val_dice:.4f}")

    print("Training Completed!")

    torch.save(model.state_dict(), "final_unet_model.pth")

    print(" Milestone 2 COMPLETE ")
    # =========================================================
# SAVE ONE FINAL IMAGE (Milestone-2)
# =========================================================
model.eval()
os.makedirs("milestone2_proof", exist_ok=True)

with torch.no_grad():
    sample_images, sample_masks = next(iter(val_loader))
    sample_images = sample_images.to(device)

    outputs = model(sample_images)
    preds = (outputs > 0.3).float()  

    # take first sample
    img_tensor = sample_images[0]
    gt = sample_masks[0].squeeze().numpy()
    pr = preds[0].cpu().squeeze().numpy()

    #  ISOLATED OUTPUT (MAIN GOAL)
    isolated_tensor = img_tensor * preds[0]

    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    isolated = isolated_tensor.cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(12,4))

    plt.subplot(1,4,1)
    plt.title("Input Image")
    plt.imshow(img)

    plt.subplot(1,4,2)
    plt.title("Ground Truth")
    plt.imshow(gt, cmap="gray")

    plt.subplot(1,4,3)
    plt.title("Predicted Mask")
    plt.imshow(pr, cmap="gray")

    plt.subplot(1,4,4)
    plt.title("Isolated Output")
    plt.imshow(isolated)

    plt.tight_layout()
    plt.savefig("milestone2_proof/final_result.png")
    plt.close()

print("One proof image with isolation saved!")