"""
Evaluation metrics for binary image segmentation.
"""

import numpy as np


def intersection_over_union(pred, target, threshold=0.5):
    pred = (pred >= threshold).astype(np.uint8)
    target = (target >= threshold).astype(np.uint8)

    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection

    if union == 0:
        return 1.0
    return intersection / (union + 1e-8)


def dice_score(pred, target, threshold=0.5):
    pred = (pred >= threshold).astype(np.uint8)
    target = (target >= threshold).astype(np.uint8)

    intersection = np.sum(pred * target)
    total = np.sum(pred) + np.sum(target)

    return (2 * intersection) / (total + 1e-8)


def accuracy_score(pred, target, threshold=0.5):
    pred = (pred >= threshold).astype(np.uint8)
    target = (target >= threshold).astype(np.uint8)

    return np.mean(pred == target)


def precision_score(pred, target, threshold=0.5):
    pred = (pred >= threshold).astype(np.uint8)
    target = (target >= threshold).astype(np.uint8)

    tp = np.sum(pred * target)
    fp = np.sum(pred * (1 - target))

    return tp / (tp + fp + 1e-8)


def recall_score(pred, target, threshold=0.5):
    pred = (pred >= threshold).astype(np.uint8)
    target = (target >= threshold).astype(np.uint8)

    tp = np.sum(pred * target)
    fn = np.sum((1 - pred) * target)

    return tp / (tp + fn + 1e-8)


def evaluate_all(pred, target, threshold=0.5):
    return {
        "IoU": intersection_over_union(pred, target, threshold),
        "Dice": dice_score(pred, target, threshold),
        "Accuracy": accuracy_score(pred, target, threshold),
        "Precision": precision_score(pred, target, threshold),
        "Recall": recall_score(pred, target, threshold),
    }