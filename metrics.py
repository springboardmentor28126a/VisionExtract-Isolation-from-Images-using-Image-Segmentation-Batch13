import torch

def dice_score(pred,target):

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()

    dice = (2 * intersection) / (pred.sum() + target.sum() + 1e-8)

    return dice


def iou_score(pred,target):

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()

    union = pred.sum() + target.sum() - intersection

    return intersection / (union + 1e-8)