import os
import random
import shutil


IMAGE_DIR = "images"
TRAIN_DIR = "images/train"
TEST_DIR = "images/test"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

images = [img for img in os.listdir(IMAGE_DIR) if img.endswith(".jpg")]

random.shuffle(images)

split_ratio = 0.8
split_index = int(len(images) * split_ratio)

train_images = images[:split_index]
test_images = images[split_index:]

for img in train_images:
    shutil.move(os.path.join(IMAGE_DIR, img), os.path.join(TRAIN_DIR, img))

for img in test_images:
    shutil.move(os.path.join(IMAGE_DIR, img), os.path.join(TEST_DIR, img))

print("Total images:", len(images))
print("Training images:", len(train_images))
print("Testing images:", len(test_images))

