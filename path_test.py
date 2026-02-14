import os

DATASET_PATH = r"E:\Datasets\COCO_FINAL"

print("Checking dataset path...")

if os.path.exists(DATASET_PATH):
    print("Dataset folder found ✅")
    print("Folders inside dataset:")
    print(os.listdir(DATASET_PATH))
else:
    print("Dataset folder NOT found ❌")
