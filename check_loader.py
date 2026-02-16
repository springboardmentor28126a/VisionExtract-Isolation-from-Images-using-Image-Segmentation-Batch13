import torch
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import CocoSegmentationDataset, get_training_augmentation

def unnormalize(tensor):
    """
    Reverses the ImageNet normalization for visualization.
    mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Move to CPU, convert to numpy, transpose (C, H, W) -> (H, W, C)
    img = tensor.permute(1, 2, 0).cpu().numpy()
    
    # Reverse formula: input = (target * std) + mean
    img = std * img + mean
    img = np.clip(img, 0, 1) # Ensure valid pixel range
    return img

def check_loader():
    # 1. Initialize Dataset with Training Augmentations (to test flips/resizes)
    dataset = CocoSegmentationDataset(
        root_dir='data/raw', 
        subset='val2017', 
        transform=get_training_augmentation()
    )
    
    # 2. Fetch a sample
    # Let's loop until we find a sample that was actually flipped/modified 
    # (though hard to know for sure, seeing one valid sample is usually enough)
    image_tensor, mask_tensor = dataset[5] # Pick an arbitrary index
    
    print(f"Image Tensor: {image_tensor.shape}")
    print(f"Mask Tensor: {mask_tensor.shape}")

    # 3. Un-normalize for display
    vis_image = unnormalize(image_tensor)
    vis_mask = mask_tensor.squeeze().cpu().numpy() # Remove channel dim

    # 4. Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(vis_image)
    ax[0].set_title("Augmented Input Image")
    ax[0].axis('off')
    
    ax[1].imshow(vis_mask, cmap='gray')
    ax[1].set_title("Augmented Binary Mask")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_loader()
