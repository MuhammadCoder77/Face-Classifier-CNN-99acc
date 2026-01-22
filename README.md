# Face Classifier CNN

## Project Overview
This project implements a **Face Classifier** using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model achieves high accuracy (99% on validation set) in distinguishing between "Face" and "Non-Face" images.

The repository includes:
- `CNN.ipynb`: Jupyter notebook for data loading, model construction, training, and evaluation.
- `predict.py`: A Python script for real-time face detection using your computer's webcam.
- Pre-trained models (`FaceClassifier.h5` and `.keras` formats).

## Features
- **Binary Classification**: Robustly classifies images as either containing a face or not.
- **Deep Learning Architecture**: Utilizes a custom 6-layer CNN (4 Convolutional layers + 2 Dense layers).
- **Real-Time Inference**: `predict.py` captures video from the webcam and runs the model on each frame to detect faces live.
- **High Accuracy**: Trained to ~99% accuracy on the validation split.

## Dataset Details
The model was trained on a synthesized dataset containing **10,000 images**:
- **Classes**: 2 (Face, Non-Face)
- **Distribution**: Balanced, with approximately 5,000 images per class.
- **Input Shape**: Images are resized to **250x250 pixels** (RGB).
- **Synthesis**: 5,000 images for the Face class were taken from the LFW dataset, whereas 1,000 images from the CIFAR-10 dataset were augmented to create 5,000 images for the Non-face class.

## Model Architecture
The core model is a Sequential CNN with the following structure:

| Layer Type | Output Shape | Param # | Description |
| :--- | :--- | :--- | :--- |
| **Input** | (250, 250, 3) | 0 | RGB Image Input |
| **Rescaling** | (250, 250, 3) | 0 | Normalizes pixel values [0, 1] |
| **Conv2D** | (250, 250, 16) | 448 | 16 filters, 3x3 kernel, ReLU |
| **MaxPooling2D**| (125, 125, 16) | 0 | 2x2 pool size |
| **Conv2D** | (123, 123, 32) | 4,640 | 32 filters, 3x3 kernel, ReLU |
| **MaxPooling2D**| (61, 61, 32) | 0 | 2x2 pool size |
| **Conv2D** | (59, 59, 64) | 18,496 | 64 filters, 3x3 kernel, ReLU |
| **MaxPooling2D**| (29, 29, 64) | 0 | 2x2 pool size |
| **Conv2D** | (27, 27, 64) | 36,928 | 64 filters, 3x3 kernel, ReLU |
| **MaxPooling2D**| (13, 13, 64) | 0 | 2x2 pool size |
| **Flatten** | (10816) | 0 | Flattens 2D features to 1D |
| **Dense** | (8) | 86,536 | Fully connected, 8 units, ReLU |
| **Dense (Output)**| (1) | 9 | Sigmoid/Linear (check implementation) |

- **Total Parameters**: **147,761** (577.19 KB)
- **Trainable Parameters**: 147,761

## Installation & Requirements

Ensure you have Python installed. You can install the required dependencies using pip:

```bash
pip install tensorflow opencv-python matplotlib numpy
```

## Usage

### Real-time Detection
To run the facial classification on your webcam feed:

```bash
python predict.py
```
*Press `q` to quit the webcam window.*

### Model Training / Analysis
To view the training process, visualization of accuracy/loss, or to retrain the model:
1. Open the notebook:
   ```bash
   jupyter notebook CNN.ipynb
   ```
2. Run the cells to load data and train the model.

## Project Structure
```
.
├── CNN.ipynb                   # Training notebook
├── predict.py                  # Real-time inference script
├── FaceClassifier.h5           # Saved model (H5 format)
├── FaceClassifier.keras        # Saved model (Keras format)
├── face_classifier.weights.h5  # Model weights
├── ProjectReport.pdf           # Project documentation
└── README.md                   # This file
```
