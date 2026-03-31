import os
import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from model import UNet
from preprocess_test2 import preprocess_inference

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet().to(device)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "best_model_main_subject.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def keep_largest_component(mask):

    mask = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]

    for i in range(2, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > largest_area:
            largest_area = stats[i, cv2.CC_STAT_AREA]
            largest_label = i

    cleaned_mask = np.zeros_like(mask)
    cleaned_mask[labels == largest_label] = 1

    return cleaned_mask


def isolate_subject(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    processed = preprocess_inference(image)

    input_tensor = torch.tensor(processed).permute(0,3,1,2).float().to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output)
        pred = (pred > 0.5).float()

    pred_mask = pred.squeeze().cpu().numpy()

    pred_mask = keep_largest_component(pred_mask)

    pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    isolated = image * np.expand_dims(pred_mask, axis=-1)

    return isolated


@app.route("/", methods=["GET", "POST"])
def index():

    result_image = None
    uploaded_image = None

    if request.method == "POST":

        file = request.files["image"]

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        uploaded_image = "/" + filepath
        file.save(filepath)

        result = isolate_subject(filepath)

        result_path = os.path.join(RESULT_FOLDER, "result.png")

        cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        result_image = "/" + result_path

    return render_template("index.html", result_image=result_image,uploaded_image=uploaded_image)


if __name__ == "__main__":
    app.run(debug=True)