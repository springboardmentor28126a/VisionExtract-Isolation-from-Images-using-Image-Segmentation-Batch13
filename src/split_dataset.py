import os
import random
import shutil
from sklearn.model_selection import train_test_split

# Paths
images_dir = "data/val2017"
output_dir = "data/processed"

# Create output folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# Get all image names
all_images = os.listdir(images_dir)

print("Total images:", len(all_images))

# First split: 70% train, 30% temp
train_imgs, temp_imgs = train_test_split(
    all_images,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

# Second split: 15% val, 15% test
val_imgs, test_imgs = train_test_split(
    temp_imgs,
    test_size=0.50,
    random_state=42
)

print("Train:", len(train_imgs))
print("Val:", len(val_imgs))
print("Test:", len(test_imgs))

# Function to copy images
def copy_images(image_list, split_name):
    for img in image_list:
        src = os.path.join(images_dir, img)
        dst = os.path.join(output_dir, split_name, img)
        shutil.copy(src, dst)

copy_images(train_imgs, "train")
copy_images(val_imgs, "val")
copy_images(test_imgs, "test")

print("Dataset split completed successfully!")
