# =========================
# IMPORTS
# =========================
import streamlit as st
from PIL import Image
import numpy as np
import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision.models.segmentation import deeplabv3_resnet50

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="VisionExtract", layout="wide")

# =========================
# CUSTOM UI (SAME)
# =========================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0a192f, #112240);
        color: white;
    }

    h1 {
        text-align: center;
        color: #ff69b4;
        font-size: 42px;
    }

    p, label {
        color: #e6f1ff;
    }

    section[data-testid="stFileUploader"] {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 2px dashed #ff69b4;
    }

    .stButton>button {
        background-color: #ff69b4;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("<h1>🎯 VisionExtract - Background Remover</h1>", unsafe_allow_html=True)
st.write("Upload an image and get **subject isolated output**")

# =========================
# SIDEBAR (SAME)
# =========================
st.sidebar.title("⚙️ Settings")

bg_option = st.sidebar.selectbox(
    "Background Color",
    ["Black", "White", "Custom"]
)

custom_color = st.sidebar.color_picker("Pick Color", "#000000")

if bg_option == "Black":
    bg_color = (0, 0, 0)
elif bg_option == "White":
    bg_color = (255, 255, 255)
else:
    custom_color = custom_color.lstrip("#")
    bg_color = tuple(int(custom_color[i:i+2], 16) for i in (0, 2, 4))

# =========================
# LOAD MODEL (IMPORTANT)
# =========================
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = deeplabv3_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    model.load_state_dict(torch.load("deeplab_resnet50.pth", map_location=device))
    model.to(device)
    model.eval()

    return model, device

model, device = load_model()

# =========================
# IMAGE DISPLAY (SAME)
# =========================
IMAGE_WIDTH = 400

def show_image(image, title):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_str = buf.getvalue()

    st.markdown(f"""
        <div style="text-align:center;">
            <h3 style="color:white;">{title}</h3>
            <img src="data:image/png;base64,{base64.b64encode(img_str).decode()}"
                 style="
                    width:{IMAGE_WIDTH}px;
                    border-radius:15px;
                    box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
                 ">
        </div>
    """, unsafe_allow_html=True)

# =========================
# PROCESS FUNCTION (YOUR MODEL)
# =========================
def process_image(image):

    image_np = np.array(image)
    h, w = image_np.shape[:2]

    # Resize like training
    img_resized = cv2.resize(image_np, (256, 256)) / 255.0

    img_tensor = torch.tensor(img_resized)\
        .permute(2, 0, 1)\
        .float()\
        .unsqueeze(0)\
        .to(device)

    with torch.no_grad():
        output = model(img_tensor)['out']

        # Resize back
        output = F.interpolate(
            output,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )

        pred = torch.sigmoid(output[0, 0]).cpu().numpy()

    # Threshold
    mask = (pred > 0.5).astype(np.uint8)

    # Apply mask (black/custom bg)
    bg = np.zeros_like(image_np)
    bg[:, :] = bg_color

    result = (
        image_np * mask[:, :, None] +
        bg * (1 - mask[:, :, None])
    ).astype(np.uint8)

    return Image.fromarray(result)

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

# =========================
# MAIN FLOW
# =========================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    # PREVIEW
    st.markdown("## 🖼️ Preview")
    show_image(image, "")

    if st.button("🚀 Remove Background"):

        with st.spinner("Processing with AI model..."):
            result = process_image(image)

        st.success("✅ Done!")

        # RESULTS
        st.markdown("## 🖼️ Results")

        col1, col2 = st.columns(2)

        with col1:
            show_image(image, "Original")

        with col2:
            show_image(result, "Segmented Output")

        # DOWNLOAD
        st.markdown("## 📥 Download Result")

        buf = io.BytesIO()
        result.save(buf, format="PNG")

        st.download_button(
            label="⬇ Download Image",
            data=buf.getvalue(),
            file_name="output.png",
            mime="image/png"
        )
