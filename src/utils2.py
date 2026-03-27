import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.utils import Sequence

# The Loss Function for segmentation
def dice_loss(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + 1.0) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.0)
    return 1 - dice

class VisionExtractAdvancedGenerator(Sequence):
    def __init__(self, data_dir, batch_size=16, img_size=(256, 256), **kwargs):
        # FIX: Add this line to satisfy Keras 3 requirements
        super().__init__(**kwargs) 
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.image_filenames = [f for f in os.listdir(data_dir) if f.endswith('_resized.jpg')]

    def __len__(self):
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = [], []

        for filename in batch_files:
            base_name = filename.replace('_resized.jpg', '')
            img_path = os.path.join(self.data_dir, filename)
            mask_path = os.path.join(self.data_dir, f"{base_name}_mask.png")

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            # --- WEEK 5 ADVANCED AUGMENTATION ---
            # Randomly apply Color Jitter or Rotation to improve robustness [cite: 74]
            choice = np.random.choice(['none', 'color', 'rotate', 'hflip'])
            if choice == 'color':
                img = tf.image.random_brightness(img, max_delta=0.2).numpy()
                img = tf.image.random_contrast(img, 0.7, 1.3).numpy()
            elif choice == 'rotate':
                angle = np.random.uniform(-5, 5)
                M = cv2.getRotationMatrix2D((self.img_size[1]//2, self.img_size[0]//2), angle, 1.0)
                img = cv2.warpAffine(img, M, self.img_size)
                mask = cv2.warpAffine(mask, M, self.img_size)
            elif choice == 'hflip':
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size) / 255.0
            mask = cv2.resize(mask, self.img_size) / 255.0
            mask = np.expand_dims(mask, axis=-1)

            X.append(img)
            y.append(mask)

        return np.array(X), np.array(y)