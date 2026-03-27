import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="VisionExtract Image Segmentation", layout="wide")

DEFAULT_MODEL_CANDIDATES = [
    "models/vision_extract_best.keras",
    "models/vision_extract_adv.keras",
    "models/vision_extract_fine_tuning.keras",
    "notebooks/deeplab_final.keras",
    "notebooks/deeplab_best_stage1.keras",
]


@st.cache_resource
def load_segmentation_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = load_model(model_path, compile=False)
    return model


def preprocess_image(image: Image.Image, target_size=(256, 256)) -> np.ndarray:
    image = image.convert("RGB")
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, target_size, interpolation=cv2.INTER_AREA)

    # normalized as [0,1], and shape (H,W,C)
    image_norm = image_resized.astype(np.float32) / 255.0
    return image_norm


def postprocess_mask(mask_pred: np.ndarray, original_size=None, threshold: float = 0.5) -> np.ndarray:
    if mask_pred.ndim == 4:
        mask_pred = np.squeeze(mask_pred, axis=0)

    if mask_pred.ndim == 3 and mask_pred.shape[-1] in (1,):
        mask_pred = np.squeeze(mask_pred, axis=-1)

    if mask_pred.ndim == 3 and mask_pred.shape[-1] > 1:
        # multiclass softmax-like output -> convert to one-hot mask (largest class)
        mask_pred = np.argmax(mask_pred, axis=-1)
        mask = (mask_pred > 0).astype(np.uint8) * 255
    else:
        mask = (mask_pred >= threshold).astype(np.uint8) * 255

    # Aggressive noise removal and hole filling
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Fill internal holes aggressively
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
    
    # Keep only the largest connected component (main subject)
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels > 1:
        largest_label = 1 + np.argmax(np.bincount(labels.flat)[1:])
        mask = (labels == largest_label).astype(np.uint8) * 255
    
    # Light opening to remove remaining edge noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_med, iterations=1)

    if original_size is not None:
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    return mask


def isolate_subject(image_np: np.ndarray, mask_np: np.ndarray, background_color=(255, 255, 255)) -> np.ndarray:
    """
    Extract subject using mask: keep masked region, replace background with white.
    Returns RGBA image for transparency support.
    """
    if mask_np.ndim == 2:
        mask_bool = mask_np > 0
    else:
        mask_bool = np.any(mask_np > 0, axis=-1)

    # Create RGBA output
    isolated = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
    
    # Copy subject (where mask is True)
    isolated[mask_bool] = np.concatenate([image_np[mask_bool], np.full((mask_bool.sum(), 1), 255, dtype=np.uint8)], axis=1)
    
    # Set background to white with full opacity
    isolated[~mask_bool] = [background_color[0], background_color[1], background_color[2], 255]
    
    return isolated


def main():
    st.title("VisionExtract Image Segmentation Demo")
    st.markdown(
        "Upload an image and run the pipeline: preprocessing -> inference -> output generation."
    )

    with st.sidebar:
        #st.header("Deployment")
        uploaded_file = st.file_uploader("Upload an image (JPEG/PNG/AVIF)", type=["jpg", "jpeg", "png", "avif"])

        st.markdown("**Select model:**")
        model_path = st.selectbox("Choose available model", DEFAULT_MODEL_CANDIDATES, index=0)
        
        model_path_input = st.text_input("Or enter custom path (optional)", value="", placeholder="e.g., models/my_model.keras")
        if model_path_input.strip():
            model_path = model_path_input.strip()

        run_button = st.button("Run inference")

    if uploaded_file is None:
        st.info("Please upload an image file to begin.")
        return

    input_image = Image.open(uploaded_file)
    original_size = input_image.size  # (width, height)
    st.subheader("Input image")
    st.image(input_image, width=300)

    if not run_button:
        st.info("Select model and click Run inference.")
        return

    try:
        model = load_segmentation_model(model_path)
    except Exception as e:
        st.error(f"Cannot load model: {e}")
        return

    st.success(f"Model loaded from: {model_path}")

    preprocessed = preprocess_image(input_image, target_size=(256, 256))
    model_input = np.expand_dims(preprocessed, axis=0)

    with st.spinner("Running model inference..."):
        predictions = model.predict(model_input)

    mask = postprocess_mask(predictions, original_size=input_image.size, threshold=0.5)
    st.subheader("Predicted mask")
    st.image(mask, clamp=True, channels="GRAY", width=300)

    orig_np = np.array(input_image.convert("RGB"))
    isolated_subject = isolate_subject(orig_np, mask, background_color=(0, 0, 0))

    st.subheader("Subject Isolation Output")
    st.image(isolated_subject, width=300)

    st.markdown("---")
    st.write("Pipeline complete. Download the isolated subject below.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download isolated subject"):
            # Convert RGBA to RGB for PNG save
            isolated_rgb = isolated_subject[:, :, :3]
            out_pil = Image.fromarray(isolated_rgb.astype(np.uint8))
            out_pil.save("isolated_subject.png")
            st.success("isolated_subject.png saved to working directory")
    
    with col2:
        if st.button("Download mask"):
            out_pil = Image.fromarray(mask)
            out_pil.save("predicted_mask.png")
            st.success("predicted_mask.png saved to working directory")


if __name__ == "__main__":
    main()
