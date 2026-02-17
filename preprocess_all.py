from dataset import COCODataset
from torch.utils.data import DataLoader

# Create dataset
dataset = COCODataset(
    annotation_file="annotations/instances_val2017.json",
    image_folder="val2017",
    image_size=512,
    transform=False
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

print("Total Images:", len(dataset))

# Loop through entire dataset
for i, (images, masks) in enumerate(dataloader):
    print(f"Processing batch {i+1}")
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)

print("✅ Preprocessing Complete for Entire val2017")
