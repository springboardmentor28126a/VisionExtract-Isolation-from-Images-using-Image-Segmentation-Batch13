"""
utils_deeplab.py
VisionExtract — Data Generators for DeepLabV3+ (CPU Optimized)
Place this file in: src/utils_deeplab.py
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────
def dice_loss(y_true, y_pred):
    """Same dice loss used in UNet — kept for comparison."""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1e-6) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-6
    )


def bce_dice_loss(y_true, y_pred):
    """
    Combined BCE + Dice loss.
    BCE handles class imbalance (many background pixels).
    Dice directly optimizes the overlap metric.
    This is why DeepLab should escape the 'all-black mask' trap your UNet hit.
    """
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice


# ─────────────────────────────────────────────────────────────────────────────
# Training Generator
# ─────────────────────────────────────────────────────────────────────────────
class DeepLabTrainGenerator(Sequence):
    """
    Loads train batches from your preprocessed folder.
    Randomly picks one augment variant per image per epoch.
    Variants used: normalized, hflip, vflip, bright, rot30
    (all already saved to disk by your preprocessing pipeline)
    """

    VARIANTS = ["normalized", "hflip", "vflip", "bright", "rot30"]

    def __init__(self, data_dir, batch_size=4, img_size=(128, 128), **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size

        self.image_ids = [
            f.replace("_normalized.jpg", "")
            for f in os.listdir(data_dir)
            if f.endswith("_normalized.jpg")
        ]
        np.random.shuffle(self.image_ids)
        print(f"[TrainGen] {len(self.image_ids)} images found in {data_dir}")

    def __len__(self):
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.image_ids)

    def __getitem__(self, index):
        batch_ids = self.image_ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = [], []

        for img_id in batch_ids:
            variant = np.random.choice(self.VARIANTS)
            img_path = os.path.join(self.data_dir, f"{img_id}_{variant}.jpg")
            mask_path = os.path.join(self.data_dir, f"{img_id}_mask.png")

            # Fallback to normalized if variant file is missing
            if not os.path.exists(img_path):
                img_path = os.path.join(self.data_dir, f"{img_id}_normalized.jpg")

            img  = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img  = cv2.resize(img,  self.img_size).astype(np.float32) / 255.0
            mask = cv2.resize(mask, self.img_size).astype(np.float32) / 255.0
            mask = np.expand_dims(mask, axis=-1)

            X.append(img)
            y.append(mask)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Validation / Test Generator  (no augmentation)
# ─────────────────────────────────────────────────────────────────────────────
class DeepLabValGenerator(Sequence):
    """Deterministic generator for val and test splits."""

    def __init__(self, data_dir, batch_size=4, img_size=(128, 128), **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size

        self.image_ids = [
            f.replace("_normalized.jpg", "")
            for f in os.listdir(data_dir)
            if f.endswith("_normalized.jpg")
        ]
        print(f"[ValGen]   {len(self.image_ids)} images found in {data_dir}")

    def __len__(self):
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        batch_ids = self.image_ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = [], []

        for img_id in batch_ids:
            img_path  = os.path.join(self.data_dir, f"{img_id}_normalized.jpg")
            mask_path = os.path.join(self.data_dir, f"{img_id}_mask.png")

            img  = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img  = cv2.resize(img,  self.img_size).astype(np.float32) / 255.0
            mask = cv2.resize(mask, self.img_size).astype(np.float32) / 255.0
            mask = np.expand_dims(mask, axis=-1)

            X.append(img)
            y.append(mask)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
class BinaryIoU(tf.keras.metrics.Metric):
    """
    Correct IoU for binary segmentation.
    Thresholds predictions at 0.5 before computing overlap.
    """
    def __init__(self, threshold=0.5, name="binary_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold    = threshold
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union        = self.add_weight(name="union",        initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true > self.threshold, tf.float32)
        intersection = tf.reduce_sum(y_pred * y_true)
        union        = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true) - intersection
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        return (self.intersection + 1e-6) / (self.union + 1e-6)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)