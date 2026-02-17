# sample_image.py

from Data_Preprocessing.preprocessing import preprocess_image
from Data_Preprocessing.img_viz import show_image
import cv2

image_path = "data/data/coco2017/train2017/000000001497.jpg"


image = preprocess_image(image_path)

# Convert back to 0-255 for visualization
image_display = (image * 255).astype("uint8")

show_image(image_display, "Preprocessed Image")







