# Pneumonia Detection from Chest X-Rays

**An end-to-end deep learning project for binary classification of pneumonia using chest X-ray images.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📌 Project Overview

This project implements and compares four different deep learning architectures to detect pneumonia (NORMAL vs PNEUMONIA) from chest X-ray images using the well-known Kaggle dataset.

**Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (~5,863 images across train, validation, and test sets)

**Models Implemented**:
- Custom CNN (baseline model with convolutional and max-pooling layers)
- EfficientNetB0 (transfer learning with ImageNet pre-trained weights)
- CNN-BiLSTM hybrid (CNN feature extractor + bidirectional LSTM treating image rows as sequences)
- TimeDistributed CNN + LSTM (RNN-style model processing image patches sequentially)

**Key Features**:
- Data preprocessing and augmentation (rotation, zoom, flip, etc.)
- Handling class imbalance
- Transfer learning with fine-tuning
- Model interpretability using Grad-CAM heatmaps
- Comprehensive evaluation (confusion matrices, ROC curves, precision-recall curves, accuracy/loss plots)
- Side-by-side model performance comparison
- Jupyter-based prediction function that loads any trained model and displays prediction, confidence score, and Grad-CAM visualization

## 🚀 Sample Visualizations

<!-- Replace these placeholders with actual paths once you upload screenshots -->
![Grad-CAM Heatmap Example](visualizations/gradcam_example.png)  
*Grad-CAM visualization highlighting regions of interest*

![Model Comparison](visualizations/model_comparison.png)  
*Performance comparison across all models*

## 🛠 Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- Matplotlib & Seaborn
- NumPy & Pandas
- Jupyter Notebook
- GPU acceleration (CUDA)

## 📂 Project Structure
├── notebooks/                  # Main Jupyter notebook(s)
├── models/                     # Saved model weights (.h5 files)
├── visualizations/             # Plots, confusion matrices, Grad-CAM images
├── data/                       # Dataset folder (download from Kaggle - not uploaded)
├── requirements.txt            # (Optional) List of dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file

