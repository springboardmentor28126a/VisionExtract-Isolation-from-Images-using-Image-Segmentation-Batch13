import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Import our custom modules from the src folder
from src.dataset import CocoSegmentationDataset, get_training_augmentation
from src.model import VisionExtractUNet

def visualize_predictions(model, dataset, device, num_samples=2):
    """
    Passes a few samples through the model to visualize its early learning.
    
    Args:
        model (torch.nn.Module): The trained neural network.
        dataset (torch.utils.data.Dataset): The dataset to pull samples from.
        device (torch.device): CPU or CUDA.
        num_samples (int): How many images to visualize.
    """
    model.eval() # Set model to evaluation mode
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    with torch.no_grad(): # No gradients needed for inference
        for i in range(num_samples):
            # Get data
            image_tensor, true_mask = dataset[i]
            
            # Add batch dimension and move to device: (C, H, W) -> (1, C, H, W)
            input_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Predict
            raw_logits = model(input_tensor)
            
            # Apply Sigmoid to convert logits to probabilities (0.0 to 1.0)
            prob_mask = torch.sigmoid(raw_logits)
            
            # Threshold to make it strictly binary (1 or 0)
            pred_mask = (prob_mask > 0.5).float()
            
            # Convert tensors back to numpy for matplotlib
            vis_img = image_tensor.permute(1, 2, 0).numpy()
            
            # Reverse ImageNet normalization for visualization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            vis_img = std * vis_img + mean
            vis_img = np.clip(vis_img, 0, 1)
            
            # Plot Original Image
            axes[i, 0].imshow(vis_img)
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis('off')
            
            # Plot Ground Truth
            axes[i, 1].imshow(true_mask.squeeze().numpy(), cmap='gray')
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis('off')
            
            # Plot Prediction
            axes[i, 2].imshow(pred_mask.squeeze().cpu().numpy(), cmap='gray')
            axes[i, 2].set_title("Model Prediction")
            axes[i, 2].axis('off')
            
    plt.tight_layout()
    plt.show()

def train_model():
    """
    Executes the training pipeline for the U-Net model using Dice Loss.
    Initializes loaders, model, optimizer, and runs the epoch loop.
    """
    # 1. Setup Device (Will use CPU since you are local, but ready for Colab GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Data Setup
    # Using a tiny batch size (2 or 4) to prevent CPU RAM overload locally
    dataset = CocoSegmentationDataset(
        root_dir='data/raw', 
        subset='val2017', 
        transform=get_training_augmentation()
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # 3. Model Setup
    model = VisionExtractUNet().to(device)

    # 4. Loss and Optimizer
    # We use mode='binary' and from_logits=True so we don't need to change model.py
    criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop
    epochs = 3
    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # tqdm gives us a nice progress bar in the CLI
        bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks in bar:
            images = images.to(device)
            masks = masks.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            
            # Calculate Dice Loss
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item()
            bar.set_postfix({"Dice Loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete | Average Dice Loss: {epoch_loss:.4f}")

    print("\nTraining complete! Visualizing results after 3 epochs...")
    visualize_predictions(model, dataset, device)

if __name__ == "__main__":
    train_model()
