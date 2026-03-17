import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Import our custom modules from the src folder
from src.dataset import CocoSegmentationDataset, get_training_augmentation, get_validation_augmentation
from src.model import VisionExtractModel

def train_model():
    """
    Executes the full training and validation pipeline.
    Saves the best model weights based on validation Dice Loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Data Setup (Now with Training and Validation sets)
    # Using larger batch sizes for GPU
    batch_size = 16 if torch.cuda.is_available() else 2
    
    train_dataset = CocoSegmentationDataset(
        root_dir='data/raw', subset='train2017', transform=get_training_augmentation()
    )
    val_dataset = CocoSegmentationDataset(
        root_dir='data/raw', subset='val2017', transform=get_validation_augmentation()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 2. Model Setup: Ultimate Run Configuration
    model = VisionExtractModel(arch="unetplusplus", encoder_name="efficientnet-b3").to(device)
    # 3. Loss, Optimizer, and Scheduler
    criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    # Dynamically reduce LR by half if Val Loss plateaus for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # 4. Training & Validation Loop
    epochs = 20 # train for 20 epochs on the GPU
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(epochs):
        # --- TRAIN PHASE ---
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # ADD THIS LINE: Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        
        # Checkpoint Saving: Save the model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = 'checkpoints/best_model_unetplusplus_effb3.pth'
            torch.save(model.state_dict(), save_path)
            print(f"--> Validation loss improved! Saved model to {save_path}")

if __name__ == "__main__":
    train_model()
