import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from model import UNet
from preprocess_test2 import preprocess_inference

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = UNet().to(device)
model.load_state_dict(torch.load("best_model_main_subject.pth", map_location=device))
model.eval()


# -----------------------------
# Post-processing
# -----------------------------
def keep_largest_component(mask):

    mask = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]

    for i in range(2, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > largest_area:
            largest_area = stats[i, cv2.CC_STAT_AREA]
            largest_label = i

    cleaned_mask = np.zeros_like(mask)
    cleaned_mask[labels == largest_label] = 1

    return cleaned_mask


# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image_path):

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess
    processed = preprocess_inference(image)

    input_tensor = torch.tensor(processed).permute(0,3,1,2).float().to(device)

    # Model prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output)
        pred = (pred > 0.5).float()

    pred_mask = pred.squeeze().cpu().numpy()

    # Post-processing
    pred_mask = keep_largest_component(pred_mask)

    kernel = np.ones((5,5), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    pred_mask = cv2.medianBlur(pred_mask, 5)

    # Resize mask to original image size
    pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Isolate subject
    isolated = image * np.expand_dims(pred_mask, axis=-1)

    # Visualization
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(isolated)
    plt.title("Isolated Subject")
    plt.axis("off")

    plt.show()


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":

    image_path = input("Enter path of unseen image: ").strip().replace('"','')
    predict_image(image_path)