# week3_unet_fast.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Dataset ----------------
class SegDataset(Dataset):

    def __init__(self, image_dir, mask_dir, size=128):

        self.images = []
        self.masks = []
        mask_files = os.listdir(mask_dir)

        for img_name in sorted(os.listdir(image_dir)):

            img = cv2.imread(os.path.join(image_dir,img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(size,size))
            img = img.astype(np.float32)/255.0
            img = np.transpose(img,(2,0,1))

            mask = np.zeros((size,size), dtype=np.float32)
            base = os.path.splitext(img_name)[0]

            for mf in mask_files:
                if mf.startswith(base):
                    m = cv2.imread(os.path.join(mask_dir,mf),0)
                    m = cv2.resize(m,(size,size))
                    mask = np.maximum(mask,m)

            mask = mask/255.0
            mask = np.expand_dims(mask,0)

            self.images.append(img)
            self.masks.append(mask)

        self.images = torch.tensor(np.array(self.images),dtype=torch.float32)
        self.masks = torch.tensor(np.array(self.masks),dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        return self.images[idx], self.masks[idx]

# ---------------- U-Net ----------------
class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU()
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(128,64,2,stride=2)

        self.dec2 = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(64,32,2,stride=2)

        self.dec1 = nn.Sequential(
            nn.Conv2d(64,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.ReLU()
        )

        self.final = nn.Conv2d(32,1,1)

    def forward(self,x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2,e2],1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1,e1],1)
        d1 = self.dec1(d1)

        return self.final(d1)

# ---------------- Metrics ----------------
def compute_iou(pred,mask):

    pred = (pred>0.5).float()

    intersection = (pred*mask).sum((1,2,3))
    union = (pred+mask - pred*mask).sum((1,2,3))

    return ((intersection+1e-6)/(union+1e-6)).mean().item()

def compute_dice(pred,mask):

    pred = (pred>0.5).float()

    inter = (pred*mask).sum((1,2,3))

    dice = (2*inter +1e-6)/(pred.sum((1,2,3))+mask.sum((1,2,3))+1e-6)

    return dice.mean().item()

# ---------------- Visualization ----------------
def visualize_predictions(model, dataloader, num_samples=3):

    model.eval()

    imgs, masks = next(iter(dataloader))
    imgs = imgs.to(device)

    with torch.no_grad():
        preds = torch.sigmoid(model(imgs))

    imgs = imgs.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()

    for i in range(num_samples):

        img = np.transpose(imgs[i], (1,2,0))
        mask = masks[i][0]
        pred = preds[i][0]

        pred = (pred > 0.5).astype(np.float32)

        plt.figure(figsize=(10,3))

        plt.subplot(1,3,1)
        plt.title("Input Image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.title("Ground Truth")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.title("Prediction")
        plt.imshow(pred, cmap="gray")
        plt.axis("off")

        plt.show()

# ---------------- Paths ----------------
image_dir = r"C:\Users\Thanuja\Desktop\vision_extraction\project_folder\images"
mask_dir  = r"C:\Users\Thanuja\Desktop\vision_extraction\project_folder\masks"

# ---------------- Dataset ----------------
dataset = SegDataset(image_dir,mask_dir)

print("Dataset size:",len(dataset))

train_size = int(0.8*len(dataset))
val_size = len(dataset)-train_size

train_ds,val_ds = random_split(dataset,[train_size,val_size])

train_loader = DataLoader(train_ds,batch_size=4,shuffle=True)
val_loader = DataLoader(val_ds,batch_size=4)

# ---------------- Model ----------------
model = UNet().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(),lr=1e-3)

# ---------------- Training ----------------
epochs = 5

print("🚀 Training started...")

for epoch in range(epochs):

    print(f"Epoch {epoch+1} running...")

    model.train()

    train_loss = 0

    for imgs,masks in train_loader:

        imgs,masks = imgs.to(device),masks.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)

        loss = criterion(outputs,masks)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---------- Validation ----------
    model.eval()


    val_iou = 0
    val_dice = 0

    with torch.no_grad():

        for imgs,masks in val_loader:

            imgs,masks = imgs.to(device),masks.to(device)

            preds = torch.sigmoid(model(imgs))

            val_iou += compute_iou(preds,masks)
            val_dice += compute_dice(preds,masks)

    val_iou /= len(val_loader)
    val_dice /= len(val_loader)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}")

print("✅ Training completed!")

print("🔍 Visualizing predictions...")
visualize_predictions(model, val_loader, num_samples=3)

print("✅ Week 3 proof completed!")