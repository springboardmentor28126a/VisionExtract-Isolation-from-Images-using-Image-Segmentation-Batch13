import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3(nn.Module):

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()

        model = deeplabv3_resnet50(
            weights="DEFAULT" if pretrained_backbone else None
        )

        # Replace classifier to output 1 channel (binary segmentation)
        in_ch = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_ch, 1, kernel_size=1)

        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out["out"]


# Alias for compatibility with existing scripts
UNet = DeepLabV3
