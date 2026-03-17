import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class VisionExtractModel(nn.Module):
    """
    Modular architecture for binary subject isolation.
    Supports U-Net, FPN, and DeepLabV3+ for experimentation.
    """
    def __init__(self, arch="unet", encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
        """
        Initializes the chosen segmentation model.
        
        Args:
            arch (str): Architecture type ('unet', 'fpn', 'deeplabv3plus').
            encoder_name (str): Backbone architecture (default: resnet34).
            encoder_weights (str): Pre-training dataset (default: imagenet).
            in_channels (int): Input channels (3 for RGB).
            classes (int): Output channels (1 for binary mask).
        """
        super(VisionExtractModel, self).__init__()
        
        self.arch = arch.lower()
        
        if self.arch == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name, encoder_weights=encoder_weights,
                in_channels=in_channels, classes=classes, activation=None
            )
        elif self.arch == "fpn":
            # Feature Pyramid Network: Excellent for detecting objects at different scales
            self.model = smp.FPN(
                encoder_name=encoder_name, encoder_weights=encoder_weights,
                in_channels=in_channels, classes=classes, activation=None
            )
        elif self.arch == "unetplusplus":
            # UNet++: Densely connected skip pathways for ultra-sharp boundaries
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name, encoder_weights=encoder_weights,
                in_channels=in_channels, classes=classes, activation=None
            )
        elif self.arch == "deeplabv3plus":
            # DeepLabV3+: Uses Atrous Spatial Pyramid Pooling for sharp boundaries
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name, encoder_weights=encoder_weights,
                in_channels=in_channels, classes=classes, activation=None
            )
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    def forward(self, x):
        return self.model(x)
