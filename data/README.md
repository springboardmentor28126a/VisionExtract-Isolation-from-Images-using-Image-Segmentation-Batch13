VisionExtract: Image Subject Isolation using Deep Learning Segmentation

## 🎯 Project Overview

A computer vision project that automatically extracts and isolates the main subject from images using advanced deep learning segmentation models. The system processes input images to generate precise binary masks and isolated subjects with clean backgrounds, making it ideal for applications in photography, digital art, virtual backgrounds, and augmented reality.

VisionExtract uses image segmentation techniques to identify and extract foreground subjects from complex backgrounds. The project implements multiple neural network architectures including DeepLabV3+ and Feature Pyramid Networks (FPN) with ResNet backbones, trained on the COCO 2017 dataset.

### Key Features

- **Advanced Segmentation Models**: Multiple pre-trained models including DeepLabV3+ and FPN architectures
- **Interactive Web Demo**: Streamlit-based web application for real-time image processing
- **Comprehensive Data Pipeline**: Automated data preprocessing, augmentation, and validation
- **Subject Isolation**: Clean extraction of main subjects with customizable background replacement
- **Batch Processing**: Support for processing multiple images efficiently
- **Model Evaluation**: Built-in metrics and visualization tools for model performance analysis

## 📁 Project Structure

```
VISIONEXTRACT/
├── LICENSE                          # MIT License
├── README.md                        # Project documentation
├── streamlit_app.py                 # Interactive web demo application
├── data/
│   ├── README.md                    # Dataset documentation
│   ├── val2017/                     # Original COCO val2017 images
│   ├── annotations/
│   │   └── instances_val2017.json   # COCO annotation file
│   └── processed/                   # Preprocessed dataset
│       ├── train/                   # Training data (70%)
│       ├── val/                     # Validation data (15%)
│       └── test/                    # Test data (15%)
├── models/                          # Pre-trained model files
│   ├── vision_extract_best.keras
│   ├── vision_extract_adv.keras
│   └── vision_extract_fine_tuning.keras
├── notebooks/                       # Jupyter notebooks for development
│   ├── dataset_exploration.ipynb    # Data analysis, visualization, preprocessing verification, unet model training and fpn model training
│   ├── deeplab_train.ipynb          # DeepLab model training
│   ├── deeplab_best_stage1.keras
│   ├── deeplab_best_stage2.keras
│   ├── deeplab_final.keras
│   └── models/
│       ├── fpn_visionextract_final.keras
│       └── fpn_visionextract_finetuned.keras
├── requirements/
│   └── requirements.txt             # Python dependencies
├── src/                             # Source code
│   ├── model.py                     # unet Model architectures
│   ├── model_deeplab.py             # DeepLab model architecture
│   ├── model1.py                    # unet adv model architecture
│   ├── utils.py                     # unet initial training
│   ├── utils_deeplab.py             # DeepLab training
│   ├── utils1.py                    # unet finetuned training
│   ├── utils2.py                    # unet adv training
│   ├── utils3.py                    # fpn training
│   ├── split_dataset.py             # Dataset splitting logic
│   └── __pycache__/                 # Python cache files

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- TensorFlow 2.x

### Installation

 **Create a virtual environment** (recommended):
   ```bash
   conda create -n visionextract python=3.8
   conda activate visionextract
   ```

 **Install dependencies**:
   ```bash
   pip install -r requirements/requirements.txt
   pip install streamlit  # Additional requirement for the web app
   ```

### Running the Web Demo

```bash
streamlit run streamlit_app.py
```

This launches an interactive web application where you can:
- Upload images (JPEG, PNG, AVIF formats)
- Select from available pre-trained models
- View preprocessing, mask prediction, and subject isolation results
- Download processed outputs

## 📊 Dataset

### Source Data
- **Primary Dataset**: COCO 2017 Validation Set
- **Annotation Format**: COCO JSON format (`instances_val2017.json`)
- **Data Split**: 70% Training, 15% Validation, 15% Test

### Preprocessing Pipeline
The data processing pipeline includes:
- **Image Resizing**: Standardized to 256x256 pixels
- **Normalization**: Pixel values scaled to [0,1] range
- **Data Augmentation**: Horizontal/vertical flips, brightness adjustments, rotations
- **Mask Generation**: Binary segmentation masks for subject isolation
- **Subject Isolation**: Clean subject extraction with black backgrounds

## 🧠 Models

### Available Models

1. **vision_extract_best.keras** - Primary production model
2. **vision_extract_adv.keras** - Advanced variant with improved accuracy
3. **vision_extract_fine_tuning.keras** - Fine-tuned model for specific use cases
4. **DeepLab Models**:
   - `deeplab_final.keras` - Final trained DeepLab model
   - `deeplab_best_stage1.keras` - Best model from first training stage
   - `deeplab_best_stage2.keras` - Best model from second training stage
5. **FPN Models**:
   - `fpn_visionextract_final.keras` - Final FPN model
   - `fpn_visionextract_finetuned.keras` - Fine-tuned FPN variant

### Model Architectures

- **DeepLabV3+**: Atrous convolution-based segmentation with encoder-decoder structure
- **Feature Pyramid Network (FPN)**: Multi-scale feature fusion for improved segmentation accuracy
- **Backbone**: Mobilenetv2
- **Loss Functions**: Dice loss, Focal loss, and combinations
- **Output**: Binary segmentation masks

## 🏋️ Training

### Training Notebooks

- `dataset_exploration.ipynb`: Data analysis, visualization, preprocessing verification, unet model training and fpn model training
- `deeplab_train.ipynb`: Complete training pipeline for DeepLab models

### Training Process

1. **Data Preparation**: Load and preprocess COCO dataset
2. **Model Initialization**: Load pre-trained backbones with ImageNet weights
3. **Training Stages**:
   - Stage 1: Initial training with frozen backbone
   - Stage 2: Fine-tuning with unfrozen backbone layers
4. **Evaluation**: Dice coefficient, IoU, precision, recall metrics
5. **Model Selection**: Best models saved based on validation performance


## 📈 Performance Metrics

The models are evaluated using standard segmentation metrics:
- **Dice Coefficient**: Measures overlap between predicted and ground truth masks
- **IoU (Intersection over Union)**: Pixel-level overlap metric
- **Accuracy**: Pixel classification accuracy on binary subject/background
- **Loss**: Dice loss used for training (lower is better)

### Per-model evaluation (one batch from `data/processed/val`, 256x256)

| Model                                                                |Accuracy|  IoU   | Dice Loss|        Notes                   |
|---|---|---|---|---|
| UNet (`models/vision_extract_best.keras`)                            | 0.7686 | 0.5558 | 0.3405 | Baseline UNet-style model        |
| Fine Tuned UNet (`models/vision_extract_fine_tuning.keras`)          | 0.7805 | 0.6116 | 0.3387 | Fine-tuned UNet variant          |
| Updated UNet (`models/vision_extract_adv.keras`)                     | 0.6148 | 0.5063 | 0.2807 | Advanced UNet with modifications |
| FPN (`notebooks/models/fpn_visionextract_final.keras`)               | 0.8057 | 0.5545 | 0.2357 | Feature Pyramid Network model    |
| Fine Tuned FPN (`notebooks/models/fpn_visionextract_finetuned.keras`)| 0.8062 | 0.5491 | 0.2512 | Fine-tuned FPN variant           |
| DeepLabV3+ Stage1 (`notebooks/deeplab_best_stage1.keras`)            | 0.8335 | 0.5884 | 0.2474 | DeepLabV3+ after stage 1 training|
| DeepLabV3+ Stage2 (`notebooks/deeplab_best_stage2.keras`)            | 0.8251 | 0.5773 | 0.2578 | DeepLabV3+ after stage 2 training|
