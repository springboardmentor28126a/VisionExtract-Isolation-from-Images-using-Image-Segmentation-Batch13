import torch
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler

from torch_dataset import SegmentationTorchDataset
from model import UNet
from losses import BCEDiceLoss


assert torch.cuda.is_available(), "CUDA GPU is required but not available."
DEVICE = torch.device("cuda")
print("Using device:", DEVICE)

dataset = SegmentationTorchDataset(
    image_dir="images/train",
    mask_dir="masks/train",
    augment=True
)
total_size = len(dataset)

train_size = int(0.7 * total_size)

val_size = int(0.15 * total_size)

test_size = total_size - train_size - val_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator
)


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


model = UNet().to(DEVICE)


criterion = BCEDiceLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.5
)


scaler = GradScaler()


EPOCHS = 20

best_val = float("inf")


for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()

    train_loss = 0


    for i, (images, masks) in enumerate(train_loader):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        optimizer.zero_grad()
        
        with autocast(device_type="cuda"):

            preds = model(images)

            loss = criterion(preds, masks)
        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        train_loss += loss.item()


        if i % 20 == 0:

            print(
                f"[Epoch {epoch+1}/{EPOCHS}] "
                f"[Batch {i}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )


    train_loss /= len(train_loader)
    model.eval()

    val_loss = 0
    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)


            with autocast(device_type="cuda"):

                preds = model(images)

                loss = criterion(preds, masks)


            val_loss += loss.item()


    val_loss /= len(val_loader)


    print(f"\nTrain Loss: {train_loss:.4f}")

    print(f"Val Loss: {val_loss:.4f}")


    if val_loss < best_val:

        best_val = val_loss

        torch.save(model.state_dict(), "best_unet.pth")

        print("Best model saved")


    scheduler.step()


print("\nTraining complete")
