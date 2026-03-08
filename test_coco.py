import os

path = "val2017"

images = os.listdir(path)

print("Total images:",len(images))
print("Sample image:",images[0])