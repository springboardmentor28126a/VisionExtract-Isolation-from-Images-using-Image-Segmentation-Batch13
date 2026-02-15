import os
import random
import shutil

# Paths
images_path = "dataset/images/val2017"
output_base = "dataset"

# Create new folders
train_path = os.path.join(output_base, "train")
val_path = os.path.join(output_base, "val")
test_path = os.path.join(output_base, "test")

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Get all images
all_images = os.listdir(images_path)
random.shuffle(all_images)

total = len(all_images)

train_split = int(0.7 * total)
val_split = int(0.85 * total)

train_images = all_images[:train_split]
val_images = all_images[train_split:val_split]
test_images = all_images[val_split:]

def copy_images(image_list, destination):
    for img in image_list:
        src = os.path.join(images_path, img)
        dst = os.path.join(destination, img)
        shutil.copy(src, dst)

copy_images(train_images, train_path)
copy_images(val_images, val_path)
copy_images(test_images, test_path)

print("Dataset split completed!")
print(f"Train: {len(train_images)}")
print(f"Val: {len(val_images)}")
print(f"Test: {len(test_images)}")
