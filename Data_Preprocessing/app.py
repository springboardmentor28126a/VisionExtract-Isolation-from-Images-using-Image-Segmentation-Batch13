import streamlit as st
import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from streamlit_lottie import st_lottie
from streamlit_image_comparison import image_comparison
import io
import numpy as np
from PIL import Image
import requests

# ==============================
# Model Loading
# ==============================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)

model.load_state_dict(torch.load("resnet50_model.pth", map_location=device))
print("Model loaded successfully.")
model.to(device)
model.eval()

# ==============================
# Preprocessing
# ==============================

transform = A.Compose([
    A.Resize(256,256)
])

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

def segment_subject(image):

    original = image.copy()

    input_img = cv2.resize(image, (512,512))
    input_img = input_img / 255.0
    input_img = (input_img - mean) / std
    input_img = np.transpose(input_img, (2,0,1))

    input_img = torch.from_numpy(input_img).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_img)

    output = output.detach().cpu()
    mask = torch.sigmoid(output)[0][0].numpy()

    mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    mask = (mask > 0.5).astype(np.float32)

    subject = original * mask[:,:,None]

    return mask, subject

# # ==============================
# # Streamlit UI
# # ==============================

# st.title("Subject Segmentation App")

# st.write("Upload an image and extract the main subject.")

# uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

# if uploaded_file is not None:

#     image = Image.open(uploaded_file)
#     image = np.array(image)

#     st.subheader("Original Image")
#     st.image(image)

#     mask, subject = segment_subject(image)

#     st.subheader("Predicted Mask")
#     st.image(mask)

#     st.subheader("Subject Extracted")
#     st.image(subject.astype(np.uint8))

#     subject_pil = Image.fromarray(subject.astype(np.uint8))

#     st.download_button(
#         label="Download Result",
#         data=subject_pil.tobytes(),
#         file_name="images.jpg"
#     )
def add_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, #1f4037, #99f2c8);
            color: white;
        }}

        .glass {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            animation: fadeIn 1s ease-in-out;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .title {{
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }}

        .subtitle {{
            text-align: center;
            font-size: 18px;
            margin-bottom: 30px;
        }}

        .stButton>button {{
           background: linear-gradient(90deg, #ff9966, #ff5e62);
    color: white !important;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            font-size: 16px;
            transition: 0.3s;
        }}

        .stButton>button:hover {{
            transform: scale(1.05);
    background: linear-gradient(90deg, #ff5e62, #ff9966);

        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg()

# ==============================
# Title Section
# ==============================

st.markdown('<div class="title"> Subject Segmentation App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered subject extraction with DeepLabV3+</div>', unsafe_allow_html=True)

# ==============================
# Upload Section
# ==============================

st.markdown('<div class="glass">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg","png","jpeg"])

st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# Processing & Display
# ==============================

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = np.array(image)

    with st.spinner("🔍 Segmenting subject..."):
        mask, subject = segment_subject(image)

    st.markdown("## ✨ Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📷 Original")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### 🧠 Mask")
        st.image(mask, use_container_width=True)

    with col3:
        st.markdown("### 🎯 Extracted")
        st.image(subject.astype(np.uint8), use_container_width=True)

    # ==============================
    # Download Button
    # ==============================

    subject_pil = Image.fromarray(subject.astype(np.uint8))

    st.download_button(
        label="⬇️ Download Result",
        data=subject_pil.tobytes(),
        file_name="segmented.png"
    )