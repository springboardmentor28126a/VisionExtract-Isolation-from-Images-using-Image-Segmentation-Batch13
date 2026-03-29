# VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13



VisionExtract is an end-to-end Machine Learning pipeline and web application designed to automatically extract the main subject from an image, rendering the background completely black. This replicates the "cutout" functionality essential in modern media editing, virtual conferencing, and automated photography pipelines.

##  Key Features
* **High-Fidelity Edge Detection**: Achieves precise pixel-level segmentation using a custom-trained **UNet++** architecture.
* **Robust Generalization**: Capable of isolating single subjects or dense crowds across varied lighting conditions and complex backgrounds.
* **Decoupled Architecture**: Features a production-ready **FastAPI** backend serving model inferences to a standalone **Vanilla JS/HTML** web frontend.
* **Real-Time Processing**: Lightweight EfficientNet backbone allows for fast CPU/GPU inference.

##  Model Architecture & Training
* **Architecture**: [UNet++] 
* **Encoder Backbone**: [EfficientNet-B3] (Pre-trained on ImageNet)
* **Dataset**: COCO 2017 (Trained on 64,000+ isolated 'person' masks)
* **Augmentation Strategy**: ShiftScaleRotate, RandomBrightnessContrast, HorizontalFlip (via Albumentations)
* **Loss Function**: Binary Cross Entropy (BCE) + Dice Loss
* **Learning Rate Optimization**: `ReduceLROnPlateau` dynamic scheduling.

###  Performance Metrics (Unseen Test Set)
* **Mean Dice Score**: `0.8710`
* **Mean Intersection over Union (IoU)**: `0.7799`

##  System Architecture

The project strictly follows a decoupled Client-Server model:
1. **Frontend (User Interface)**: A lightweight HTML/JS drag-and-drop web application.
2. **Backend (REST API)**: A FastAPI server that receives image blobs, processes them through the PyTorch model, and returns the isolated output.


##  Tech Stack
* **Deep Learning**: PyTorch, Segmentation Models PyTorch (SMP)
* **Computer Vision**: OpenCV, Albumentations
* **Backend API**: FastAPI, Uvicorn, Python-Multipart
* **Frontend**: HTML5, CSS3, JavaScript (Vanilla)

##  Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/springboardmentor28126a/VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13.git](https://github.com/springboardmentor28126a/VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13.git)
cd VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13
```

### 2. Install Dependencies
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### 3. Run the Backend (FastAPI Server)
Start the inference engine. It will expose a REST endpoint at `http://0.0.0.0:8000/extract`.
```bash
python api.py
```

### 4. Run the Frontend (Web UI)
Open a new terminal window, navigate to the frontend directory, and start a local HTTP server:
```bash
cd frontend
python -m http.server 8080
```
Navigate to `http://localhost:8080` in your web browser to use the application!

##  Project Structure
```text
VisionExtract/
├── api.py                 # FastAPI backend server
├── train.py               # Main training script (Kaggle/Colab)
├── evaluate.py            # Official metrics evaluation script
├── infer.py               # CLI tool for single image testing
├── frontend/              # Web application client
│   ├── index.html
│   └── script.js
├── src/                   # Core ML architecture modules
│   ├── dataset.py
│   ├── model.py
│   └── postprocess.py
├── checkpoints/           # Trained model weights (.pth)
└── scripts/               # Data acquisition and exploration tools
```

##  Author
**Rishikesh P.**
*Completed as part of the Springboard AI/ML Mentorship Program.*
```

