import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn as nn

# -----------------------------
# SIMPLE MODEL (same as before)
# -----------------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",   # try this first
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(torch.load("week5_coco_model.pth", map_location=device))

model.to(device)
model.eval()
# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🧠 Image Subject Extraction App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8)

    # Resize mask
    mask = cv2.resize(mask, image.size)

    image_np = np.array(image)

    # Apply mask
    result = image_np * mask[:, :, None]

    st.image(result, caption="Output (Background Removed)", use_column_width=True)