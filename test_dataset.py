from dataset import SegmentationDataset

dataset = SegmentationDataset("val2017")

print("Dataset size:",len(dataset))
image,mask = dataset[0]

print("Image shape:",image.shape)
print("Mask shape:",mask.shape)