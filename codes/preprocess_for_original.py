import cv2
import matplotlib.pyplot as plt
from preprocess_test2 import preprocess_inference

def test_sample(image_path):

    # Load image properly
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess
    processed = preprocess_inference(image)

    # Remove batch dimension for visualization
    processed_display = processed[0]

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(processed_display)
    plt.title("Preprocessed Image (Resized + Normalized)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_sample(r"C:\VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13\VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13\data\pre.jpg")