import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from src.model import VisionExtractUNet
from src.dataset import CocoSegmentationDataset, get_validation_augmentation

def load_model(weights_path, device):
    """Loads the trained U-Net model."""
    model = VisionExtractUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # Set to evaluation mode!
    return model

def unnormalize(tensor):
    """Reverses ImageNet normalization for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    return np.clip(img, 0, 1)

def visualize_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 1. Load the Model
    weights_path = "checkpoints/best_model.pth"
    try:
        model = load_model(weights_path, device)
        print("Successfully loaded model weights.")
    except FileNotFoundError:
        print(f"Error: Could not find {weights_path}. Did you download it from Colab?")
        return

    # 2. Load the Validation Dataset
    dataset = CocoSegmentationDataset(
        root_dir='data/raw', 
        subset='val2017', 
        transform=get_validation_augmentation()
    )

    # 3. Pick a random image
    idx = random.randint(0, len(dataset) - 1)
    image_tensor, true_mask = dataset[idx]

    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # 4. Generate Prediction
    with torch.no_grad():
        raw_logits = model(input_tensor)
        prob_mask = torch.sigmoid(raw_logits)
        pred_mask = (prob_mask > 0.5).float() # Threshold at 50%

    # 5. Visualize
    vis_img = unnormalize(image_tensor)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(vis_img)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    axes[1].imshow(true_mask.squeeze().numpy(), cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask.squeeze().cpu().numpy(), cmap='gray')
    axes[2].set_title("Model Prediction")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_prediction()
