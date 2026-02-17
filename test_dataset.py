from dataset import COCODataset
import matplotlib.pyplot as plt

dataset = COCODataset(
    annotation_file="annotations/instances_val2017.json",
    image_folder="val2017",
    image_size=512,
    transform=False
)

image, mask = dataset[0]

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Preprocessed Image")
plt.imshow(image.permute(1,2,0))
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Preprocessed Mask")
plt.imshow(mask.squeeze(0), cmap="gray")
plt.axis("off")

plt.show()
