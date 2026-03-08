import os
import random
import shutil


source_folder = "C:/Users/varsh/Downloads/archive/coco2017/val2017"
destination_folder = "C:/Users/varsh/Downloads/archive/coco2017/split_data"


# Create destination folders
train_folder = os.path.join(destination_folder, "train")
val_folder = os.path.join(destination_folder, "val")
test_folder = os.path.join(destination_folder, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get all images
all_images = os.listdir(source_folder)

# Shuffle images
random.shuffle(all_images)

# Total count
total_images = len(all_images)

# Split ratios
train_count = int(0.7 * total_images)
val_count = int(0.15 * total_images)
test_count = total_images - train_count - val_count

# Split images
train_images = all_images[:train_count]
val_images = all_images[train_count:train_count + val_count]
test_images = all_images[train_count + val_count:]

# Function to copy images
def copy_images(image_list, destination):
    for img in image_list:
        src_path = os.path.join(source_folder, img)
        dst_path = os.path.join(destination, img)
        shutil.copy(src_path, dst_path)

# Copy files
copy_images(train_images, train_folder)
copy_images(val_images, val_folder)
copy_images(test_images, test_folder)

print("✅ Dataset successfully split!")
print(f"Train: {len(train_images)}")
print(f"Validation: {len(val_images)}")
print(f"Test: {len(test_images)}")
