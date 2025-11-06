# Fruits and Vegetables Recognition System

## 1. Overview

The **Fruits and Vegetables Recognition System** is a deep learning–based image classification project developed to automatically identify various fruits and vegetables from digital images.
The project employs a **Convolutional Neural Network (CNN)** architecture implemented in **TensorFlow** and **Keras**, demonstrating the practical application of deep learning in computer vision for agricultural and retail domains.

---

## 2. Objectives

1. To design and train a convolutional neural network capable of classifying multiple categories of fruits and vegetables.
2. To evaluate the model’s performance using standard metrics and validate its generalization capability.
3. To develop a scalable framework applicable to real-world scenarios such as automated food sorting, smart retail systems, and agricultural monitoring.

---

## 3. Methodology

### 3.1 Dataset Preparation

1. A custom dataset was organized into **training**, **validation**, and **test** subsets.
2. Each image was resized to **64 × 64 pixels**, converted to **RGB format**, and normalized to ensure uniform input representation.
3. The dataset was loaded and preprocessed using TensorFlow’s `image_dataset_from_directory()` function, enabling efficient data batching and augmentation.

### 3.2 Model Architecture

1. The CNN model was implemented using **TensorFlow** and **Keras** frameworks.
2. The architecture consists of multiple **convolutional layers** with **ReLU** activation functions for spatial feature extraction.
3. **Max pooling layers** were applied to reduce spatial dimensionality while retaining key visual features.
4. **Dropout layers** were integrated to prevent overfitting and enhance model generalization.
5. **Fully connected dense layers** were used for classification.
6. The **softmax output layer** contained **36 units**, corresponding to the number of fruit and vegetable categories.

### 3.3 Training and Optimization

1. **Optimizer:** Adam
2. **Loss Function:** Categorical Cross-Entropy
3. **Regularization Techniques:** Batch Normalization and Dropout
4. The model was trained for multiple epochs until validation performance convergence was achieved.

---

## 4. Evaluation

| Metric              | Accuracy |
| :------------------ | :------: |
| Training Accuracy   |  0.8501  |
| Validation Accuracy |  0.8632  |
| Test Accuracy       |  0.8579  |

The results indicate that the model achieved consistent and balanced performance across all datasets, demonstrating strong generalization and minimal overfitting.

---

## 5. Tools and Libraries

1. Python
2. TensorFlow / Keras
3. NumPy
4. Matplotlib
5. Google Colab

---

## 6. Results and Discussion

The developed CNN model successfully classifies fruit and vegetable images with high accuracy and robust generalization. The use of normalization and regularization techniques contributed to stable convergence and reduced overfitting.

Future work may focus on:

1. Expanding the dataset to include additional fruit and vegetable categories and greater image diversity.
2. Implementing **transfer learning** with pre-trained architectures such as **ResNet** or **MobileNet** to further improve performance.
3. Deploying the model as a **web or mobile application** for real-time classification and user interaction.

