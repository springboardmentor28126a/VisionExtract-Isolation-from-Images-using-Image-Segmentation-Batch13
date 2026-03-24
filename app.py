import numpy as np
import gradio as gr
import cv2
import torch
from PIL import Image

from infer import load_model, postprocess_mask, DEVICE
from preprocess import IMG_SIZE, imagenet_normalize


model = load_model()


def process_image(image: Image.Image | np.ndarray):
 
    if image is None:
        return None

    # Gradio input can be PIL or numpy depending on configuration
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure RGB (drop alpha if present)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Keep original size for the final output
    orig_h, orig_w = image.shape[:2]
    orig_float = image.astype(np.float32) / 255.0

    # Resize to training size and normalize like in training for the model
    resized = cv2.resize(orig_float, (IMG_SIZE, IMG_SIZE))
    resized = imagenet_normalize(resized)

    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if DEVICE == "cuda":
            # Faster inference on GPU; keeps accuracy much closer than manual half().
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(tensor)
        else:
            logits = model(tensor)

    # Soft alpha at network resolution in [0,1]
    alpha_small = postprocess_mask(logits)  # [IMG_SIZE, IMG_SIZE] float

    # Resize alpha back to original image resolution
    alpha = cv2.resize(
        alpha_small.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # Black background composite (RGB)
    extracted = orig_float * alpha[:, :, None]
    extracted = np.clip(extracted * 255, 0, 255).astype(np.uint8)
    return extracted


demo = gr.Interface(
    fn=process_image,
    # Using PIL improves upload/preview compatibility across formats and browsers
    inputs=gr.Image(type="pil", image_mode="RGB", label="Upload image"),
    outputs=gr.Image(type="numpy", image_mode="RGB", label="Subject-isolated output"),
    title="Subject Isolation",
    description="Upload an image. The model will keep only the main subject and black out the background.",
)


if __name__ == "__main__":
    demo.launch()

