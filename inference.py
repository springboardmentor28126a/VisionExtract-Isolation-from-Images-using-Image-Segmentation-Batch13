import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp

# -------------------------------
# 1. LOAD MODEL (U-Net ResNet34)
# -------------------------------
def load_model(model_path):

    model = smp.Unet(
        encoder_name="resnet34",     # ✅ as you specified
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    # Load trained weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    model.eval()
    return model


# -------------------------------
# 2. PREPROCESS IMAGE
# -------------------------------
def preprocess_image(image_path):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    return input_tensor, image


# -------------------------------
# 3. PREDICT MASK
# -------------------------------
def predict(model, input_tensor):

    with torch.no_grad():
        output = model(input_tensor)   # ✅ U-Net returns tensor

        output = torch.sigmoid(output)
        mask = output.squeeze().numpy()

    return mask


# -------------------------------
# 4. APPLY MASK (SUBJECT ISOLATION)
# -------------------------------
def apply_mask(original_image, mask):

    # 👉 adjust threshold if needed
    mask = (mask > 0.3).astype(np.uint8)

    # Resize mask to original size
    mask_img = Image.fromarray(mask * 255).resize(original_image.size)

    original_np = np.array(original_image)
    mask_np = np.array(mask_img)

    # Apply mask
    result = original_np * (mask_np[:, :, None] // 255)

    return Image.fromarray(result.astype(np.uint8)), mask_img


# -------------------------------
# 5. SAVE OUTPUTS
# -------------------------------
def save_outputs(result, mask_img, output_path, mask_path):

    result.save(output_path)
    mask_img.save(mask_path)

    print(f"✅ Output saved at: {output_path}")
    print(f"✅ Mask saved at: {mask_path}")


# -------------------------------
# 6. FULL PIPELINE
# -------------------------------
def run_pipeline(model_path, input_image_path, output_path, mask_path):

    print("Model exists:", os.path.exists(model_path))
    print("Image exists:", os.path.exists(input_image_path))

    model = load_model(model_path)

    input_tensor, original_image = preprocess_image(input_image_path)

    mask = predict(model, input_tensor)

    result, mask_img = apply_mask(original_image, mask)

    save_outputs(result, mask_img, output_path, mask_path)


# -------------------------------
# 7. MAIN
# -------------------------------
if __name__ == "__main__":

    # 👉 MODEL PATH (Week 5)
    model_path = "week5_coco_model.pth"

    # 👉 INPUT IMAGE (new unseen image)
    input_image_path = "C:\\Users\\Thanuja\\Downloads\\INDIA-11571-5dd24338d9de0__880.jpg"

    # 👉 OUTPUT IMAGE
    output_path = "output.png"

    # 👉 MASK IMAGE
    mask_path = "mask.png"

    run_pipeline(model_path, input_image_path, output_path, mask_path)