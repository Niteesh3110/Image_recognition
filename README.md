# Tomato Plant Factory Dataset - Image Recognition AI

This repository contains a Jupyter Notebook that implements a Convolutional Neural Network (CNN) for image recognition on the Tomato Plant Factory Dataset. The goal is to predict the count of tomatoes in images using a deep learning model.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Visualizations](#visualizations)
7. [Results](#results)
8. [Requirements](#requirements)
9. [Usage](#usage)

## Introduction

This project focuses on building a CNN model to predict the number of tomatoes in images from the Tomato Plant Factory Dataset. The model is trained on preprocessed images and their corresponding labels, which indicate the count of tomatoes.

## Dataset

The dataset consists of images and corresponding label files. Each image is preprocessed by resizing and normalizing pixel values. The labels are extracted from text files, where each non-empty line represents a tomato.

### Dataset Structure

- **Images Directory**: Contains `.JPG` and `.PNG` images.
- **Labels Directory**: Contains `.txt` files with tomato counts.

## Model Architecture

The CNN model is built using TensorFlow and Keras. The architecture includes:

- Two Conv2D layers with ReLU activation.
- Two MaxPooling2D layers.
- A Flatten layer.
- Two Dense layers with ReLU and linear activation respectively.

### Model Summary

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear'),
])
```
## Requirements

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tomato-plant-factory.git
   cd tomato-plant-factory
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```
