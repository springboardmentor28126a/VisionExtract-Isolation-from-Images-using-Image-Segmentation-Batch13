import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# =====================================
# PARAMETERS
# =====================================

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 8
EPOCHS = 10
LR = 0.0001

# PATHS (Using Student Name: Thanuja)

image_path = r"C:/Users/Thanuja/Desktop/vision_extraction/project_folder/images/"
mask_path = r"C:/Users/Thanuja/Desktop/vision_extraction/project_folder/masks/"
model_path = r"C:/Users/Thanuja/Desktop/vision_extraction/models/unet_model.h5"
output_path = r"C:/Users/Thanuja/Desktop/vision_extraction/outputs/predictions/"

os.makedirs(output_path, exist_ok=True)

# =====================================
# LOAD DATA
# =====================================

def load_data(img_dir, mask_dir):

    images = []
    masks = []

    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(img_files, mask_files):

        img = cv2.imread(os.path.join(img_dir, img_file))
        img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
        img = img/255.0

        mask = cv2.imread(os.path.join(mask_dir, mask_file),0)
        mask = cv2.resize(mask,(IMG_WIDTH,IMG_HEIGHT))
        mask = mask/255.0
        mask = np.expand_dims(mask,axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

X,y = load_data(image_path,mask_path)

print("Images shape:",X.shape)
print("Masks shape:",y.shape)

# =====================================
# TRAIN VALIDATION SPLIT
# =====================================

X_train,X_val,y_train,y_val = train_test_split(
    X,y,test_size=0.2,random_state=42
)

print("Train:",X_train.shape)
print("Validation:",X_val.shape)

# =====================================
# LOAD MODEL
# =====================================

from tensorflow.keras import layers, models

def build_unet():

    inputs = layers.Input((128,128,3))

    c1 = layers.Conv2D(16,3,activation='relu',padding='same')(inputs)
    c1 = layers.Conv2D(16,3,activation='relu',padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32,3,activation='relu',padding='same')(p1)
    c2 = layers.Conv2D(32,3,activation='relu',padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64,3,activation='relu',padding='same')(p2)

    u4 = layers.UpSampling2D()(c3)
    u4 = layers.concatenate([u4,c2])
    c4 = layers.Conv2D(32,3,activation='relu',padding='same')(u4)

    u5 = layers.UpSampling2D()(c4)
    u5 = layers.concatenate([u5,c1])
    c5 = layers.Conv2D(16,3,activation='relu',padding='same')(u5)

    outputs = layers.Conv2D(1,1,activation='sigmoid')(c5)

    model = models.Model(inputs,outputs)

    return model

model = build_unet()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Training model...")

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=8,
    validation_data=(X_val,y_val)
)

model.save("models/unet_model.h5")

print("Model saved successfully!")

# =====================================
# METRICS
# =====================================

def dice_coef(y_true,y_pred):

    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    intersection = tf.reduce_sum(y_true*y_pred)

    return (2*intersection+1)/(tf.reduce_sum(y_true)+tf.reduce_sum(y_pred)+1)

def iou(y_true,y_pred):

    intersection = tf.reduce_sum(y_true*y_pred)
    union = tf.reduce_sum(y_true)+tf.reduce_sum(y_pred)-intersection

    return (intersection+1)/(union+1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="binary_crossentropy",
    metrics=[dice_coef,iou]
)

# =====================================
# PREDICTIONS
# =====================================

preds = model.predict(X_val)

pred_masks = (preds>0.5).astype(np.uint8)

# =====================================
# SAVE OUTPUT IMAGES
# =====================================

for i in range(len(pred_masks)):

    pred = pred_masks[i].squeeze()*255
    filename = f"prediction_{i}.png"

    cv2.imwrite(os.path.join(output_path,filename),pred)

print("Predictions saved to:",output_path)

# =====================================
# VISUALIZATION
# =====================================

plt.figure(figsize=(12,6))

for i in range(3):

    plt.subplot(3,3,i*3+1)
    plt.imshow(X_val[i])
    plt.title("Image")
    plt.axis("off")

    plt.subplot(3,3,i*3+2)
    plt.imshow(y_val[i].squeeze(),cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(3,3,i*3+3)
    plt.imshow(pred_masks[i].squeeze(),cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

plt.tight_layout()
plt.show()

# =====================================
# MODEL EVALUATION
# =====================================

results = model.evaluate(X_val,y_val)

print("\nValidation Results")
print("Loss:",results[0])
print("Dice:",results[1])
print("IoU:",results[2])

# =====================================
# DATA AUGMENTATION
# =====================================

image_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

mask_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

seed=1

img_generator = image_gen.flow(X_train,batch_size=BATCH_SIZE,seed=seed)
mask_generator = mask_gen.flow(y_train,batch_size=BATCH_SIZE,seed=seed)

train_generator = zip(img_generator,mask_generator)

# =====================================
# FINE TUNING
# =====================================

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train)//BATCH_SIZE,
    validation_data=(X_val,y_val),
    epochs=EPOCHS
)

# =====================================
# SAVE MODEL
# =====================================

model.save(r"C:/Users/Thanuja/Desktop/vision_extraction/models/unet_model_finetuned.h5")

print("Fine tuned model saved successfully!")