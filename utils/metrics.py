"""
Segmentation metrics for binary subject isolation.

All functions operate on numpy arrays of shape (H, W) with values in {0, 1}.
"""
import numpy as np
from typing import Dict


def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray,
              threshold: float = 0.5) -> float:
    """
    Intersection over Union (Jaccard Index).
    pred_mask: float array (sigmoid output) or binary.
    """
    pred_bin = (pred_mask >= threshold).astype(np.uint8)
    true_bin = (true_mask >= threshold).astype(np.uint8)

    intersection = np.logical_and(pred_bin, true_bin).sum()
    union        = np.logical_or(pred_bin, true_bin).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def dice_coefficient(pred_mask: np.ndarray, true_mask: np.ndarray,
                     threshold: float = 0.5, smooth: float = 1.0) -> float:
    """
    Dice / F1 score.
    """
    pred_bin = (pred_mask >= threshold).astype(np.uint8).flatten()
    true_bin = (true_mask >= threshold).astype(np.uint8).flatten()

    intersection = (pred_bin * true_bin).sum()
    return float(2.0 * intersection + smooth) / \
           float(pred_bin.sum() + true_bin.sum() + smooth)


def pixel_accuracy(pred_mask: np.ndarray, true_mask: np.ndarray,
                   threshold: float = 0.5) -> float:
    """Fraction of correctly classified pixels."""
    pred_bin = (pred_mask >= threshold).astype(np.uint8)
    true_bin = (true_mask >= threshold).astype(np.uint8)
    return float((pred_bin == true_bin).mean())


def precision_recall(pred_mask: np.ndarray, true_mask: np.ndarray,
                     threshold: float = 0.5):
    """Returns (precision, recall)."""
    pred_bin = (pred_mask >= threshold).astype(np.uint8).flatten()
    true_bin = (true_mask >= threshold).astype(np.uint8).flatten()

    tp = int((pred_bin * true_bin).sum())
    fp = int((pred_bin * (1 - true_bin)).sum())
    fn = int(((1 - pred_bin) * true_bin).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    return float(precision), float(recall)


def compute_all_metrics(pred_mask: np.ndarray,
                        true_mask: np.ndarray,
                        threshold: float = 0.5) -> Dict[str, float]:
    """Compute IoU, Dice, PixelAcc, Precision, Recall in one call."""
    prec, rec = precision_recall(pred_mask, true_mask, threshold)
    return {
        "iou":       iou_score(pred_mask, true_mask, threshold),
        "dice":      dice_coefficient(pred_mask, true_mask, threshold),
        "pixel_acc": pixel_accuracy(pred_mask, true_mask, threshold),
        "precision": prec,
        "recall":    rec,
    }


class MetricTracker:
    """Accumulates per-batch metrics and returns epoch averages."""

    def __init__(self):
        self._sums   = {}
        self._counts = {}

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self._sums[k]   = self._sums.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0)  + 1

    def averages(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k]
                for k in self._sums}

    def reset(self):
        self._sums.clear()
        self._counts.clear()
