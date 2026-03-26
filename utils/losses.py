"""
Loss functions for binary segmentation.

CombinedLoss = α·BCE + β·Dice
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Operates on raw logits (applies sigmoid internally).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits)
        # Flatten spatial dims
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / \
               (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum of BCEWithLogitsLoss and DiceLoss.
    Default weights from config (0.5 / 0.5).
    """

    def __init__(
        self,
        bce_weight: float  = config.BCE_WEIGHT,
        dice_weight: float = config.DICE_WEIGHT,
    ):
        super().__init__()
        self.bce_w  = bce_weight
        self.dice_w = dice_weight
        self.bce    = nn.BCEWithLogitsLoss()
        self.dice   = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce  = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_w * loss_bce + self.dice_w * loss_dice
