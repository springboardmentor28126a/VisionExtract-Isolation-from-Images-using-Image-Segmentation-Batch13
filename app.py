import streamlit as st
import torch
import cv2
import numpy as np
from model import get_model

st.title("Object Isolation using Image Segmentation")

device = torch.device("cpu")

model = get_model()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    image = cv2.resize(image, (256,256))
    img_input = image / 255.0

    img_tensor = torch.tensor(img_input).permute(2,0,1).unsqueeze(0).float()

    with torch.no_grad():
        output = model(img_tensor)
        mask = torch.sigmoid(output).squeeze().numpy()

    # 🔥 BETTER THRESHOLD
    binary_mask = (mask > 0.5).astype(np.uint8)

    # 🔥 SMOOTH MASK
    kernel = np.ones((5,5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 🔥 CLEAN EXTRACTION
    isolated = cv2.bitwise_and(image, image, mask=binary_mask)

    st.image(image, caption="Original Image", channels="BGR")
    st.image(mask, caption="Predicted Mask")
    st.image(binary_mask*255, caption="Binary Mask")
    st.image(isolated, caption="Isolated Object", channels="BGR")

    st.success("✅ Segmentation Completed Successfully!")