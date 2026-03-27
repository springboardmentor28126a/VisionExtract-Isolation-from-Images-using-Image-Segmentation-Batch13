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
# -------- DATA AUGMENTATION --------

augmented_images = []

# 1. Horizontal Flip
flip_h = cv2.flip(normalized, 1)
augmented_images.append(("Horizontal Flip", flip_h))

# 2. Vertical Flip
flip_v = cv2.flip(normalized, 0)
augmented_images.append(("Vertical Flip", flip_v))

# 3. Rotation (15 degrees)
h, w = normalized.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 15, 1.0)
rotated = cv2.warpAffine(normalized, M, (w, h))
augmented_images.append(("Rotated 15°", rotated))

# 4. Brightness Adjustment
bright = np.clip(normalized * 1.2, 0, 1)
augmented_images.append(("Brightened", bright))
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(normalized)
plt.title("Original (Preprocessed)")
plt.axis("off")

for i, (title, img) in enumerate(augmented_images):
    plt.subplot(2,3,i+2)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
