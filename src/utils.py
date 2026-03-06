import numpy as np
import cv2
import os
import tensorflow as tf

class VisionExtractGenerator(tf.keras.utils.Sequence):
    """
    Custom Data Generator for the VisionExtract project.
    Loads images and masks in batches to manage memory efficiency.
    """
    def __init__(self, data_dir, batch_size=8, img_size=(256, 256), **kwargs):
        # The super().__init__ call removes the PyDataset warning
        super().__init__(**kwargs)
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Identify all normalized images in the directory
        self.image_filenames = [f for f in os.listdir(data_dir) if f.endswith('_normalized.jpg')]

    def __len__(self):
        # Returns the number of batches per epoch
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        # Generates one batch of data
        batch_filenames = self.image_filenames[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = [], []
        
        for filename in batch_filenames:
            # Load and process original image
            img_path = os.path.join(self.data_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size) / 255.0
            
            # Load and process corresponding mask
            mask_filename = filename.replace('_normalized.jpg', '_mask.png')
            mask_path = os.path.join(self.data_dir, mask_filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size) / 255.0
            mask = np.expand_dims(mask, axis=-1)
            
            X.append(img)
            y.append(mask)
            
        return np.array(X), np.array(y)

def dice_loss(y_true, y_pred):
    """
    Dice Loss function to handle pixel-level subject isolation accuracy.
    Formula: 1 - (2 * intersection) / (sum of pixels)
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    
    # 1e-6 added for numerical stability to avoid division by zero
    return 1 - (2. * intersection + 1e-6) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-6)