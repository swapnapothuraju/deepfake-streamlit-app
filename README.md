# 🕵️‍♂️ Deepfake Detection using Deep Learning (MobileNetV2)

## 📌 Project Overview
This project focuses on detecting deepfake images using deep learning techniques. Deepfakes are synthetically generated or manipulated images that can be used to spread misinformation and cause security concerns. The goal of this project is to accurately classify images as **Real** or **Fake** using a Convolutional Neural Network with transfer learning.

---

## 🎯 Objectives
- Detect whether an image is real or deepfake
- Build an efficient and lightweight deep learning model
- Apply transfer learning for better accuracy
- Evaluate model performance using standard metrics

---

## 🛠️ Technologies Used
- Python  
- TensorFlow & Keras  
- MobileNetV2 (Transfer Learning)  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## 📂 Dataset
- Image-based deepfake dataset
- Binary classification:
  - **Real Images**
  - **Fake Images**
- Data loaded and preprocessed using `ImageDataGenerator`

---

## 🧠 Model Architecture
- Base Model: **MobileNetV2** (pre-trained on ImageNet)
- Custom classification head added
- Output layer with **sigmoid activation**
- Loss function: **Binary Crossentropy**
- Optimizer: **Adam**

---

## ⚙️ Training & Evaluation
- Dataset split into training and validation sets
- Data augmentation applied to reduce overfitting
- Model evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## 📊 Results
- Achieved high accuracy in distinguishing real and fake images
- Model shows strong generalization due to transfer learning
- Lightweight architecture suitable for real-world deployment

---

## 🔍 Features
- Image-level deepfake detection
- Single image prediction support
- Visualization of confusion matrix and performance metrics



