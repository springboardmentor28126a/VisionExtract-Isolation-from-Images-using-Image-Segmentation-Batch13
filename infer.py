import argparse
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import our modular architecture
from src.model import VisionExtractModel

def get_inference_transform():
    """
    Returns the transformation pipeline for single-image inference.
    Neural networks expect inputs to exactly match their training distribution.
    This resizes to 320x320, normalizes using ImageNet stats, and converts to Tensor.
    """
    return A.Compose([
        A.Resize(height=320, width=320),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def load_trained_model(weights_path, arch, encoder_name, device):
    """
    Instantiates the model architecture and loads the pre-trained weights.
    
    Args:
        weights_path (str): Path to the .pth file.
        arch (str): The model architecture (e.g., 'unetplusplus').
        encoder_name (str): The backbone (e.g., 'efficientnet-b3').
        device (torch.device): CPU or CUDA.
        
    Returns:
        torch.nn.Module: The model in evaluation mode.
    """
    model = VisionExtractModel(arch=arch, encoder_name=encoder_name).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # Critical: Disables dropout and batch norm tracking
    return model

def extract_subject(image_path, model, device, output_path="isolated_output.png"):
    """
    Processes a custom image, isolates the subject, and saves the result.
    
    Args:
        image_path (str): Path to the local input image.
        model (torch.nn.Module): The loaded VisionExtract model.
        device (torch.device): Execution device.
        output_path (str): Where to save the final isolated image.
    """
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    # 1. Load Image (OpenCV loads as BGR, convert to RGB for model)
    original_image_bgr = cv2.imread(image_path)
    original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
    
    # Save original dimensions so we can restore the mask to the native resolution
    original_h, original_w = original_image_rgb.shape[:2]

    # 2. Preprocess
    transform = get_inference_transform()
    augmented = transform(image=original_image_rgb)
    input_tensor = augmented['image'].unsqueeze(0).to(device) # Add batch dimension -> (1, 3, 320, 320)

    # 3. Predict
    print("Running inference...")
    with torch.no_grad():
        raw_logits = model(input_tensor)
        prob_mask = torch.sigmoid(raw_logits)
        # Threshold at 50% certainty, remove batch/channel dims, move to CPU numpy
        pred_mask = (prob_mask > 0.5).float().squeeze().cpu().numpy() 

    # 4. Post-process
    # Resize the 320x320 mask back up to the original photo's resolution.
    # We use INTER_NEAREST to ensure the mask remains strictly binary (0 or 1).
    resized_mask = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    # Expand 2D mask (H, W) to 3D (H, W, 3) so it can multiply with the RGB channels
    mask_3d = np.stack([resized_mask]*3, axis=-1)
    
    # Isolate subject: Multiply original image by the binary mask (background becomes 0/black)
    isolated_subject = original_image_rgb * mask_3d
    
    # 5. Save the output
    # Convert back to BGR for OpenCV saving
    isolated_subject_bgr = cv2.cvtColor(isolated_subject.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, isolated_subject_bgr)
    print(f"\nSuccess! Isolated subject saved locally to: {os.path.abspath(output_path)}")

    # 6. Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(resized_mask, cmap='gray')
    axes[1].set_title("Generated Mask")
    axes[1].axis('off')
    
    axes[2].imshow(isolated_subject.astype(np.uint8))
    axes[2].set_title("Isolated Subject (RGB)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """CLI entry point for testing custom images."""
    parser = argparse.ArgumentParser(description="VisionExtract Command Line Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file")
    # Defaulting to your ultimate Week 5 champion configuration
    parser.add_argument("--weights", type=str, default="checkpoints/best_model_unetplusplus_effb3.pth", help="Path to model weights")
    parser.add_argument("--arch", type=str, default="unetplusplus", help="Model architecture used")
    parser.add_argument("--encoder", type=str, default="efficientnet-b3", help="Backbone encoder used")
    parser.add_argument("--output", type=str, default="isolated_output.png", help="Filename for the saved output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing inference on: {device}")
    
    print(f"Loading {args.arch.upper()} with {args.encoder.upper()} weights from {args.weights}...")
    model = load_trained_model(args.weights, args.arch, args.encoder, device)
    
    print(f"Processing {args.image}...")
    extract_subject(args.image, model, device, args.output)

if __name__ == "__main__":
    main()
