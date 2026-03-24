import io
import cv2
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import your modular architecture
from src.model import VisionExtractModel

# Initialize the FastAPI app
app = FastAPI(title="VisionExtract REST API")

# CRITICAL: Enable CORS. This allows our separate Web Application frontend 
# to communicate with this backend server securely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARCH = "unetplusplus"
ENCODER = "efficientnet-b3"
WEIGHTS_PATH = "checkpoints/best_model_unetplusplus_effb3.pth"

print(f"Loading {ARCH.upper()} model on {DEVICE}...")
model = VisionExtractModel(arch=ARCH, encoder_name=ENCODER).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval() # Set to evaluation mode

transform = A.Compose([
    A.Resize(height=320, width=320),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

@app.post("/extract")
async def extract_subject(file: UploadFile = File(...)):
    """
    Receives an Image as a Blob via HTTP Request, isolates the subject, 
    and returns the resultant Image as a Blob via HTTP Response.
    """
    # 1. Read the uploaded image bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    
    # Decode to OpenCV format and convert to RGB
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = img_rgb.shape[:2]

    # 2. Image Encoder / Preprocessing
    augmented = transform(image=img_rgb)
    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

    # 3. Image Segmentation Model Inference
    with torch.no_grad():
        raw_logits = model(input_tensor)
        prob_mask = torch.sigmoid(raw_logits)
        pred_mask = (prob_mask > 0.5).float().squeeze().cpu().numpy()

    # 4. Generate New Image with Black Background and Subject
    resized_mask = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    mask_3d = np.stack([resized_mask]*3, axis=-1)
    
    isolated_subject = img_rgb * mask_3d
    
    # 5. Prepare HTTP Response
    # Convert back to BGR for encoding to PNG
    isolated_subject_bgr = cv2.cvtColor(isolated_subject.astype(np.uint8), cv2.COLOR_RGB2BGR)
    success, encoded_img = cv2.imencode('.png', isolated_subject_bgr)
    
    if not success:
        return Response(status_code=500, content="Failed to encode image")

    # Return the resultant image as a binary blob
    return Response(content=encoded_img.tobytes(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    # Launch the REST API on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
