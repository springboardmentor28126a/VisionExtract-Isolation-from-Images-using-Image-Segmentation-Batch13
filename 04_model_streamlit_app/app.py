# =========================
# IMPORTS
# =========================
import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image

# =========================
# PAGE CONFIG (COLORFUL UI)
# =========================
st.set_page_config(
    page_title="VisionExtract",
    page_icon="🎯",
    layout="wide"
)

# =========================
# CUSTOM CSS (COLORFUL)
# =========================
st.markdown("""
    <style>
    body {
        background-color: #0f172a;
        color: white;
    }
    .main {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        color: #38bdf8;
        text-align: center;
    }
    .stButton>button {
        background-color: #38bdf8;
        color: black;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🎯 VisionExtract - Background Removal App")
st.write("Upload an image and get **subject isolated output** instantly!")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = deeplabv3_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    checkpoint = torch.load("deeplab_checkpoint.pth", map_location=device)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()

    return model, device

model, device = load_model()

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

# =========================
# PROCESS FUNCTION
# =========================
def process_image(image):
    original = np.array(image)
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    h, w = original.shape[:2]

    # Resize
    resized = cv2.resize(original, (256, 256)) / 255.0

    tensor = torch.tensor(resized)\
        .permute(2, 0, 1)\
        .float()\
        .unsqueeze(0)\
        .to(device)

    with torch.no_grad():
        output = model(tensor)['out']

        output = F.interpolate(
            output,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )

        pred = torch.sigmoid(output[0, 0]).cpu().numpy()

    # =========================
    # POST PROCESSING
    # =========================
    pred = cv2.GaussianBlur(pred, (9, 9), 0)

    threshold = 0.25 * pred.max()
    mask = (pred >= threshold).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    result = original * mask[:, :, None]
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return original, result

# =========================
# DISPLAY
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("🖼️ Preview")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    if st.button("🚀 Remove Background"):

        with st.spinner("Processing... ⏳"):
            original, result = process_image(image)

        with col2:
            st.image(result, caption="✨ Background Removed", use_container_width=True)

        # =========================
        # DOWNLOAD BUTTON
        # =========================
        result_pil = Image.fromarray(result)

        st.download_button(
            label="📥 Download Output",
            data=result_pil.tobytes(),
            file_name="output.png",
            mime="image/png"
        )

        st.success("✅ Done!")