VisionExtract – Subject Isolation using Image Segmentation

Project Description

VisionExtract is a deep learning project designed to automatically isolate the main subject from an image using image segmentation techniques.

Given an input image, the system predicts a binary segmentation mask that identifies the subject region. The output image retains the subject while converting the background pixels to black.

This project demonstrates the complete machine learning pipeline for an image segmentation system including:

- Dataset acquisition and exploration
- Data preprocessing and preparation
- Deep learning model development
- Training and prediction generation
- Evaluation using segmentation metrics

Dataset Used: COCO 2017 Dataset

---

Milestone 1 – Data Preprocessing

Objective

The goal of this milestone is to prepare the dataset for training the segmentation model. Raw images and masks must be processed to ensure consistency and correct alignment.

Proper preprocessing ensures the segmentation model learns meaningful features and performs accurate subject extraction.

Key Tasks

- Inspect dataset structure and annotations
- Preprocess images and segmentation masks
- Resize images to a consistent dimension
- Normalize image pixel values
- Convert masks into binary subject/background format
- Shuffle dataset samples for better model generalization

Implementation Files

preprocessing.py

Responsible for:

- Image resizing
- Image normalization
- Mask formatting
- Preparing input tensors for training

shuffling.py

Responsible for:

- Randomizing dataset samples
- Preventing training bias
- Improving model generalization

Expected Output

The preprocessing pipeline produces:

- Clean and aligned image–mask pairs
- Consistent image resolution
- Properly formatted binary masks
- Randomized dataset ready for training

---

Milestone 2 – Model Training

Objective

The goal of this milestone is to build and train a deep learning segmentation model capable of predicting the subject region in an image.

The architecture used for this project is based on U-Net, a popular convolutional neural network for pixel-level image segmentation.

Key Tasks

- Implement U-Net architecture
- Train the segmentation model on the processed dataset
- Generate predicted segmentation masks
- Perform early evaluation on validation data
- Prepare the model for further tuning and improvement

Implementation Files

unet_model.py

Defines the structure of the U-Net segmentation network, including:

- Encoder layers
- Decoder layers
- Skip connections
- Convolution operations

This file constructs the full segmentation model.

unet_implementation.py

Handles the execution pipeline including:

- Model initialization
- Training loop
- Loss computation
- Optimizer configuration
- Prediction generation

Model Output

The trained model generates:

- Predicted segmentation masks
- Subject-isolated output images

The output mask is applied to the original image to retain only the subject.

---

Evaluation Metrics

The segmentation performance will be evaluated using the following metrics:

Intersection over Union (IoU)

Measures the overlap between predicted mask and ground-truth mask.

Dice Coefficient

Measures similarity between predicted and actual segmentation regions.

Pixel Accuracy

Measures the percentage of correctly classified pixels.

Higher values indicate better segmentation performance.

---

Project Directory Structure

VisionExtract_Segmentation
│
├── data_preprocessing
│   │
│   ├── preprocessing.py
│   │   └── Handles image preprocessing, resizing, normalization, and mask preparation
│   │
│   └── shuffling.py
│       └── Randomizes dataset samples for training
│
├── model_training
│   │
│   ├── unet_model.py
│   │   └── Defines the U-Net segmentation architecture
│   │
│   └── unet_implementation.py
│       └── Executes model training and prediction pipeline
│
├── results
│   │
│   |── preprocessing_results
│   |    ├── results_2.png
│   |    ├── result_2.png
│   |    ├──results_3.png
│   |    ├──results_4.png
│   |    ├──results_5.png
|   |──segmentation_output.png
|   |──training_output.png
|
└── README.md



