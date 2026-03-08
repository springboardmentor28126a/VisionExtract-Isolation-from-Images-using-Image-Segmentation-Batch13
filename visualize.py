import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import get_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(num_classes=1)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()


img = cv2.imread("val2017/000000000139.jpg")
img = cv2.resize(img,(256,256))

img_input = img / 255.0
img_tensor = torch.tensor(img_input).permute(2,0,1).unsqueeze(0).float().to(device)


with torch.no_grad():
    output = model(img_tensor)

mask = torch.sigmoid(output).cpu().squeeze().numpy()

binary_mask = (mask > 0.5).astype(np.uint8)

extracted = img * binary_mask[:,:,None]


plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,4,2)
plt.title("Predicted Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1,4,3)
plt.title("Binary Mask")
plt.imshow(binary_mask, cmap="gray")
plt.axis("off")

plt.subplot(1,4,4)
plt.title("Extracted Object")
plt.imshow(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()