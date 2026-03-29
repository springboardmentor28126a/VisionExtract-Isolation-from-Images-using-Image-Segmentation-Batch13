import gradio as gr
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.model import VisionExtractModel

# 1. Load the Model Globally (so it doesn't reload on every single click)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading Ultimate Model...")
model = VisionExtractModel(arch="unetplusplus", encoder_name="efficientnet-b3").to(device)
model.load_state_dict(torch.load("checkpoints/best_model_unetplusplus_effb3.pth", map_location=device))
model.eval()

def get_transform():
    return A.Compose([
        A.Resize(height=320, width=320), # (Or 768 if you want to keep the high-res hack!)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# 2. The Core Prediction Function for Gradio
def predict(input_image):
    # Safety Check: If the user clicks submit without an image
    if input_image is None:
        return None, None

    # FIX 1: Handle RGBA (PNGs with transparency) and Grayscale images
    if len(input_image.shape) == 2:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    elif input_image.shape[-1] == 4:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)

    original_h, original_w = input_image.shape[:2]
    
    # Preprocess
    transform = get_transform()
    augmented = transform(image=input_image)
    input_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        raw_logits = model(input_tensor)
        prob_mask = torch.sigmoid(raw_logits)
        pred_mask = (prob_mask > 0.5).float().squeeze().cpu().numpy()
        
    # Post-process back to original size
    resized_mask = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    mask_3d = np.stack([resized_mask]*3, axis=-1)
    
    # Isolate subject
    isolated_subject = input_image * mask_3d
    
    # FIX 2: Force STRICT uint8 typing so Gradio doesn't panic
    final_mask = (mask_3d * 255).astype(np.uint8)
    final_subject = isolated_subject.astype(np.uint8)
    
    return final_mask, final_subject
# 3. Define the Web UI Layout
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Original Photo"),
    outputs=[
        gr.Image(type="numpy", label="AI Generated Mask"),
        gr.Image(type="numpy", label="Final Isolated Subject")
    ],
    title="VisionExtract AI - Subject Isolation",
    description="Week 6 Milestone: U-Net++ with EfficientNet-B3. Drag and drop any photo to instantly extract the human subject.",
    theme="default"
)

if __name__ == "__main__":
    # This launches a local web server!
    interface.launch()
