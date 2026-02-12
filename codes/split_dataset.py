import json
import random
import os

# Correct relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
ANNOTATION_FILE = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")
OUTPUT_DIR = os.path.join(DATA_DIR, "splits")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSON
with open(ANNOTATION_FILE, 'r') as f:
    coco = json.load(f)

# Get all image IDs
image_ids = [img["id"] for img in coco["images"]]

print("Total images:", len(image_ids))

# Shuffle
random.seed(42)
random.shuffle(image_ids)

# Split ratios
train_split = int(0.70 * len(image_ids))
val_split = int(0.85 * len(image_ids))

train_ids = image_ids[:train_split]
val_ids = image_ids[train_split:val_split]
test_ids = image_ids[val_split:]

print("Train:", len(train_ids))
print("Val:", len(val_ids))
print("Test:", len(test_ids))

# Save splits
def save_ids(filename, ids):
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        for id in ids:
            f.write(str(id) + "\n")

save_ids("train_ids.txt", train_ids)
save_ids("val_ids.txt", val_ids)
save_ids("test_ids.txt", test_ids)

print("Splits saved in data/splits/")
