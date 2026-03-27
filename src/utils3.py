import numpy as np
import cv2
import os
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

class VisionExtractFineTuneGenerator(Sequence):
    def __init__(self, data_path, batch_size=8, img_size=(256, 256)):
        self.data_path = os.path.abspath(data_path)
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Filter for JPGs that are NOT masks
        self.image_files = [f for f in os.listdir(self.data_path) 
                            if f.endswith('.jpg') and '_mask' not in f]
        
        if not self.image_files:
            raise FileNotFoundError(f"No JPG images found in {self.data_path}")

    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = [], []

        for file_name in batch_files:
            # 1. Load Image
            img = cv2.imread(os.path.join(self.data_path, file_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            
            # 2. Match Mask
            mask_name = file_name.replace('.jpg', '_mask.png')
            mask = cv2.imread(os.path.join(self.data_path, mask_name), cv2.IMREAD_GRAYSCALE)
            
            if mask is not None:
                mask = cv2.resize(mask, self.img_size)
                mask = (mask > 127).astype(np.float32)
            else:
                mask = np.zeros(self.img_size, dtype=np.float32)

            X.append(img)
            y.append(mask)

        return resnet_preprocess(np.array(X, dtype=np.float32)), np.array(y, dtype=np.float32)[..., np.newaxis]