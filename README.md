# Fruits-and-Vegetables-Recognition-System
**Overview**

The Fruits and Vegetables Recognition System is a deep learning-based image classification project designed to automatically identify various fruits and vegetables from digital images. The project utilizes a Convolutional Neural Network (CNN) built with TensorFlow and Keras, demonstrating the practical application of deep learning in computer vision for agricultural and retail contexts.

**Objectives**

To design and train a convolutional neural network capable of classifying multiple categories of fruits and vegetables.

To evaluate the model’s performance using appropriate metrics and validate its generalization capability.

To provide a scalable framework for real-world applications such as automated food sorting, smart retail systems, and agricultural monitoring.

**Methodology**
Dataset Preparation

A custom dataset was organized into training, validation, and test subsets.

All images were resized to 64 × 64 pixels, converted to RGB format, and normalized for consistent input representation.

The dataset was loaded and preprocessed using TensorFlow’s image_dataset_from_directory() function to facilitate efficient batching and augmentation.

Model Architecture

The proposed CNN model was implemented using TensorFlow and Keras, consisting of:

Multiple convolutional layers with ReLU activation for spatial feature extraction.

Max pooling layers to reduce spatial dimensionality while preserving critical features.

Dropout layers for regularization and overfitting prevention.

Fully connected dense layers for classification.

A softmax output layer with 36 units representing the number of fruit and vegetable categories.

Training and Optimization

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Regularization: Batch Normalization and Dropout

The model was trained for multiple epochs until performance convergence was achieved on validation data.

**Evaluation**
Metric	Accuracy
Training Accuracy	0.8501
Validation Accuracy	0.8632
Test Accuracy	0.8579

The results indicate that the model achieved strong and consistent performance across all datasets, demonstrating balanced generalization and minimal overfitting.

**Tools and Libraries
**
Python

TensorFlow / Keras

NumPy

Matplotlib

Google Colab

**Results and Discussion
**
The developed CNN model successfully classifies fruit and vegetable images with high accuracy and robust generalization. The use of regularization techniques and normalization contributed significantly to stable learning.
Further work could involve:

Expanding the dataset to include additional categories and image variations.

Incorporating transfer learning with pre-trained architectures such as ResNet or MobileNet.

Deploying the trained model as a web or mobile application for real-time recognition.
