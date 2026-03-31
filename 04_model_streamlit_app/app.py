# =========================
# IMPORTS
# =========================
import streamlit as st
<<<<<<< HEAD
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
=======
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
>>>>>>> f838bcf6c0f48bbbd006df0e5e11d27cb35e96b5
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
<<<<<<< HEAD
st.title("🎯 VisionExtract - Background Removal App")
st.write("Upload an image and get **subject isolated output** instantly!")

# =========================
# LOAD MODEL
=======
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
>>>>>>> f838bcf6c0f48bbbd006df0e5e11d27cb35e96b5
# =========================
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = deeplabv3_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

<<<<<<< HEAD
    checkpoint = torch.load("deeplab_checkpoint.pth", map_location=device)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
=======
    model.load_state_dict(torch.load("deeplab_resnet50.pth", map_location=device))
    model.to(device)
>>>>>>> f838bcf6c0f48bbbd006df0e5e11d27cb35e96b5
    model.eval()

    return model, device

model, device = load_model()

# =========================
<<<<<<< HEAD
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
=======
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
>>>>>>> f838bcf6c0f48bbbd006df0e5e11d27cb35e96b5
        .permute(2, 0, 1)\
        .float()\
        .unsqueeze(0)\
        .to(device)

    with torch.no_grad():
<<<<<<< HEAD
        output = model(tensor)['out']

=======
        output = model(img_tensor)['out']

        # Resize back
>>>>>>> f838bcf6c0f48bbbd006df0e5e11d27cb35e96b5
        output = F.interpolate(
            output,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )

        pred = torch.sigmoid(output[0, 0]).cpu().numpy()

<<<<<<< HEAD
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
=======
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
>>>>>>> f838bcf6c0f48bbbd006df0e5e11d27cb35e96b5
