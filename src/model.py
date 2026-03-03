import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class VisionExtractUNet(nn.Module):
    """
    U-Net based architecture for binary subject isolation.
    Utilizes a pre-trained ResNet34 encoder for robust feature extraction.
    """
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
        """
        Initializes the U-Net model.
        
        Args:
            encoder_name (str): The backbone architecture (default: resnet34).
            encoder_weights (str): Pre-training dataset (default: imagenet).
            in_channels (int): Number of input channels (3 for RGB images).
            classes (int): Number of output channels (1 for binary mask).
        """
        super(VisionExtractUNet, self).__init__()
        
        # Instantiate the U-Net model from SMP
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None 
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Raw logits of the segmentation mask of shape (B, 1, H, W).
        """
        return self.model(x)

if __name__ == "__main__":
    # --- Sanity Check ---
    # We simulate a batch of 2 RGB images, sized 320x320
    print("Initializing VisionExtractUNet...")
    dummy_model = VisionExtractUNet()
    
    dummy_input = torch.randn(2, 3, 320, 320)
    print(f"Feeding input tensor of shape: {dummy_input.shape}")
    
    # Forward pass
    output = dummy_model(dummy_input)
    print(f"Output tensor shape: {output.shape}")
    
    # Verify the output shape matches our expectations (Batch Size, Channels, Height, Width)
    assert output.shape == (2, 1, 320, 320), f"Shape mismatch! Expected (2, 1, 320, 320), got {output.shape}"
    print("SUCCESS: Model architecture is sound and ready for training.")
