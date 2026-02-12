import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---- Image Path ----
img_path = r"E:\Infosys_Springboard_Internship\COCO_Dataset(25 GB)\coco2017\train2017\000000003867.jpg"

# 1. Read image
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Resize (common size for models)
target_size = (512, 512)
resized = cv2.resize(image_rgb, target_size)

# 3. Denoising (Gaussian Blur)
denoised = cv2.GaussianBlur(resized, (5, 5), 0)

# 4. Contrast Enhancement using CLAHE
# Convert to LAB color space
lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)

# Apply CLAHE on L channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l_clahe = clahe.apply(l)

# Merge back
lab_clahe = cv2.merge((l_clahe, a, b))
enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

# 5. Normalize (0–1 range)
normalized = enhanced / 255.0

# ---- Display Original vs Preprocessed ----
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(normalized)
plt.title("Preprocessed Image")
plt.axis("off")

plt.tight_layout()
plt.show()
