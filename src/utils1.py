import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence

# 1. The Loss Function 
def dice_loss(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    # Adding a small epsilon for stability
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    
    # We focus strictly on the overlap to pull the model out of the 'black mask' trap
    dice = (2. * intersection + 1.0) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.0)
    return 1 - dice

# 2. The Fine-Tuning Data Generator
class VisionExtractFineTuneGenerator(Sequence):
    def __init__(self, data_dir, batch_size=16, img_size=(256, 256), **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        # Looking for your specific resized files
        self.image_filenames = [f for f in os.listdir(data_dir) if f.endswith('_resized.jpg')]

    def __len__(self):
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = [], []

        for filename in batch_files:
            base_name = filename.replace('_resized.jpg', '')
            choice = np.random.choice(['resized', 'bright', 'hflip'])
            
            # Use your existing augmented files
            if choice == 'bright':
                img_path = os.path.join(self.data_dir, f"{base_name}_bright.jpg")
            elif choice == 'hflip':
                img_path = os.path.join(self.data_dir, f"{base_name}_hflip.jpg")
            else:
                img_path = os.path.join(self.data_dir, filename)

            mask_path = os.path.join(self.data_dir, f"{base_name}_mask.png")

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Fallback logic to keep batch size consistent at 16
            if img is None or mask is None:
                img = cv2.imread(os.path.join(self.data_dir, filename))
                mask = cv2.imread(os.path.join(self.data_dir, f"{base_name}_mask.png"), cv2.IMREAD_GRAYSCALE)
                if img is None or mask is None: continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size) / 255.0
            mask = cv2.resize(mask, self.img_size) / 255.0
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)

            X.append(img)
            y.append(mask)

        return np.array(X), np.array(y)