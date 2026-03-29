import gc
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
import segmentation_models_pytorch as smp
import optuna
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.dataset import CocoSegmentationDataset
from src.model import VisionExtractModel

# --- 1. Tuning-Specific Augmentations (320x320 for speed) ---
# We define these locally so we don't interfere with the 512x512 production dataset.py
def get_tuning_train_aug():
    return A.Compose([
        A.Resize(height=320, width=320),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.1), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_tuning_val_aug():
    return A.Compose([
        A.Resize(height=320, width=320),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# --- 2. Global Data Setup (Prevents reloading COCO JSON every trial) ---
print("Initializing Datasets for Optuna Tuning...")
# Instantiate base datasets with distinct transforms to prevent Data Leakage
train_base = CocoSegmentationDataset(root_dir='data/raw', subset='val2017', transform=get_tuning_train_aug())
val_base = CocoSegmentationDataset(root_dir='data/raw', subset='val2017', transform=get_tuning_val_aug())

dataset_size = len(train_base)
indices = list(range(dataset_size))
np.random.seed(42) # Fixed seed ensures the exact same images are in train/val across all trials
np.random.shuffle(indices)
split = int(np.floor(0.2 * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]

# Create mutually exclusive subsets
train_dataset = Subset(train_base, train_indices)
val_dataset = Subset(val_base, val_indices)

# Batch size 16 is safe at 320x320 on a 16GB P100 GPU
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

# --- 3. Optuna Objective ---
def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    model = VisionExtractModel(arch="unetplusplus", encoder_name="efficientnet-b3").to(device)
    criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
    
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    # Learning Rate Scheduler
    epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # AMP Scaler
    scaler = GradScaler()

    # Wrap in try...finally to ensure GPU memory is cleared even if pruned
    try:
        for epoch in range(epochs):
            # --- TRAIN PHASE ---
            model.train()
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            # --- VALIDATION PHASE ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step() # Step scheduler per epoch
            
            # Report to Optuna
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return avg_val_loss

    finally:
        # THE FIX: Memory Leak Prevention
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    # Optimize for minimum Dice Loss
    study = optuna.create_study(direction="minimize")
    print("Starting Optuna Study...")
    study.optimize(objective, n_trials=15)
    
    print("\n==============================")
    print("BEST TRIAL FOUND:")
    trial = study.best_trial
    print(f"  Best Validation Dice Loss: {trial.value:.4f}")
    print("  Optimal Parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("==============================\n")
