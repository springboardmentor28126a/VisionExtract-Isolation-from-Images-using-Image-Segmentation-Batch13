import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
from torch.amp import autocast, GradScaler # Updated modern AMP syntax

# Import our custom modules
from src.dataset import CocoSegmentationDataset, get_training_augmentation, get_validation_augmentation
from src.model import VisionExtractModel

def train_model():
    """
    Executes the V2.0 Universal Subject Extractor training pipeline.
    Utilizes 512x512 resolution, Optuna-tuned hyperparameters, AMP, and Gradient Accumulation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Hardware & Accumulation Configuration
    # To survive a 16GB P100 GPU at 512x512 with UNet++/EfficientNet-B3:
    # We use a physical batch of 4, but update weights every 4 steps (Virtual Batch = 16)
    physical_batch_size = 4
    accumulation_steps = 4 

    # 2. Data Setup (Massive train2017 dataset)
    print("Loading datasets (this may take a moment)...")
    train_dataset = CocoSegmentationDataset(
        root_dir='data/raw', subset='train2017', transform=get_training_augmentation()
    )
    val_dataset = CocoSegmentationDataset(
        root_dir='data/raw', subset='val2017', transform=get_validation_augmentation()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=physical_batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=physical_batch_size, shuffle=False, num_workers=2)

    # 3. Model Setup
    model = VisionExtractModel(arch="unetplusplus", encoder_name="efficientnet-b3").to(device)
    
    # 4. Optuna-Optimized Hyperparameters
    criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
    
    # Injected from the best Optuna Trial
    OPT_LR = 0.0002833105206772683
    OPT_WEIGHT_DECAY = 1.6587538697422894e-05
    optimizer = optim.Adam(model.parameters(), lr=OPT_LR, weight_decay=OPT_WEIGHT_DECAY) 
    
    # Scheduler: Drop LR by half if validation loss plateaus for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Initialize modern PyTorch mixed precision scaler
    scaler = GradScaler('cuda')

    # 5. Training & Validation Loop
    epochs = 20 # Full production run
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    print(f"\nStarting Universal Extractor training for {epochs} epochs...")

    for epoch in range(epochs):
        # --- TRAIN PHASE ---
        model.train()
        train_loss = 0.0
        optimizer.zero_grad() # Zero gradients at the START of the epoch
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for i, (images, masks) in enumerate(train_bar):
            images, masks = images.to(device), masks.to(device)
            
            # AMP Autocast context for forward pass
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
                # Normalize the loss by accumulation steps so the gradients scale correctly
                loss = loss / accumulation_steps 
            
            # Scaled Backward pass
            scaler.scale(loss).backward()
            
            # Accumulation Logic: Only step the optimizer every 'accumulation_steps' batches
            # OR if it is the very last batch of the epoch
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            # Multiply back by accumulation_steps just for accurate logging display
            train_loss += (loss.item() * accumulation_steps)
            train_bar.set_postfix({"Loss": f"{(loss.item() * accumulation_steps):.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                
                # AMP is highly beneficial during validation speed too
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                val_loss += loss.item()
                val_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        # Step scheduler based on validation performance
        scheduler.step(avg_val_loss)

        # Checkpoint Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = 'checkpoints/best_model_v2_universal.pth'
            torch.save(model.state_dict(), save_path)
            print(f"--> Validation loss improved! Saved model to {save_path}")

if __name__ == "__main__":
    train_model()
