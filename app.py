import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image
import os

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="VisionExtract", layout="wide")

# ---------------------------
# Title & Description
# ---------------------------
st.title("🎯 VisionExtract - AI Subject Isolation")

st.markdown("""
### 📌 How It Works
1. Upload an image  
2. AI model detects objects (person, animals, fruits)  
3. A segmentation mask is created  
4. Background is removed automatically  

➡️ Powered by **U-Net (ResNet34)** semantic segmentation model  
""")

# ---------------------------
# Load Model
# ---------------------------
device = torch.device("cpu")

model_path = "milestone3_model.pth"

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Add milestone3_model.pth")
    st.stop()

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

st.success("✅ Model Loaded Successfully")

# ---------------------------
# Upload Section
# ---------------------------
st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = np.array(image)

    # Layout columns
    col1, col2, col3 = st.columns(3)

    # ---------------------------
    # Original Image
    # ---------------------------
    with col1:
        st.subheader("🖼 Original")
        st.image(image)

    # ---------------------------
    # Processing
    # ---------------------------
    with st.spinner("Processing Image..."):

        img = cv2.resize(image, (256, 256))
        img = img / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0)

        with torch.no_grad():
            pred = model(img)

        pred = torch.sigmoid(pred)
        pred = (pred > 0.3).float()

        output = img * pred

        mask = pred.squeeze().numpy()
        result = output.squeeze().permute(1, 2, 0).numpy()

    # ---------------------------
    # Mask
    # ---------------------------
    with col2:
        st.subheader("🧠 AI Mask")
        st.image(mask, clamp=True)

    # ---------------------------
    # Output
    # ---------------------------
    with col3:
        st.subheader("✨ Final Output")
        st.image(result)

    # ---------------------------
    # Download Button
    # ---------------------------
    result_img = (result * 255).astype(np.uint8)

    st.download_button(
        label="⬇ Download Result",
        data=cv2.imencode('.png', result_img)[1].tobytes(),
        file_name="output.png",
        mime="image/png"
    )

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("🚀 Built using PyTorch + Streamlit | Milestone 3 Model")