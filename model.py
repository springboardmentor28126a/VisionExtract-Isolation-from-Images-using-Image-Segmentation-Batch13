import torch
import torch.nn as nn
import torchvision.models as models

class SegmentationModel(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()

        backbone = models.resnet18(weights="DEFAULT")

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        self.decoder = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        x = torch.nn.functional.interpolate(
            x,
            size=(256,256),
            mode="bilinear",
            align_corners=False
        )

        return x


def get_model(num_classes=1):
    return SegmentationModel(num_classes)