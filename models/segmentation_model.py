"""
Segmentation model factory using segmentation_models_pytorch (smp).

Supported architectures (set via config.ARCHITECTURE):
  - "Unet"          : Classic U-Net with skip connections
  - "UnetPlusPlus"  : U-Net++ (nested dense skip paths)
  - "DeepLabV3Plus" : DeepLab v3+ (ASPP + encoder-decoder)
  - "FPN"           : Feature Pyramid Network decoder
  - "PAN"           : Pyramid Attention Network

Default: Unet + ResNet-50 + ImageNet weights.
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import config


def build_model(
    architecture: str = config.ARCHITECTURE,
    encoder: str       = config.ENCODER,
    encoder_weights: str = config.ENCODER_WEIGHTS,
    num_classes: int   = config.NUM_CLASSES,
) -> nn.Module:
    """
    Build and return a segmentation model.

    Returns a model whose forward() produces logits of shape (B, 1, H, W).
    Apply sigmoid + threshold at inference time.
    """
    arch = architecture.lower()

    kwargs = dict(
        encoder_name        = encoder,
        encoder_weights     = encoder_weights,
        in_channels         = 3,
        classes             = num_classes,
        activation          = None,   # raw logits; loss handles sigmoid
    )

    if arch == "unet":
        model = smp.Unet(**kwargs)
    elif arch == "unetplusplus":
        model = smp.UnetPlusPlus(**kwargs)
    elif arch == "deeplabv3plus":
        model = smp.DeepLabV3Plus(**kwargs)
    elif arch == "fpn":
        model = smp.FPN(**kwargs)
    elif arch == "pan":
        model = smp.PAN(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                         "Choose from: Unet, UnetPlusPlus, DeepLabV3Plus, FPN, PAN")

    return model


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load a saved model from a .pth checkpoint."""
    model = build_model()
    state = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dict and wrapped {"model_state": ...}
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[Model] Loaded checkpoint: {checkpoint_path}")
    return model
