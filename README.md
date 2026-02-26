# 🧠 Brain Tumor Classification using ResNet50

## 📌 Project Overview

This project is a Deep Learning model that classifies brain MRI images into four tumor categories using ResNet50 (Transfer Learning).

The model predicts one of the following classes:

🧬 Glioma

🧠 Meningioma

🚫 No Tumor

🔵 Pituitary Tumor

## 📸 Demo
### 🏠 Home Page
![Home](images/home.png)

### 🔍 Prediction Result
![Prediction](images/prediction.png)

### 📊 Prediction Probabilities
![Probabilities](images/probabilities.png)


## 🧠 Model Architecture

Transfer Learning using ResNet50

Custom Fully Connected Layers

Softmax activation for multi-class classification

Image preprocessing & augmentation

## ⚙️ Training Details

Optimizer: Adam

Loss: Categorical Crossentropy

Output Layer: 4 neurons (Softmax)

Evaluation Metric: Accuracy

Model saved as: brain_tumor_resnet50.keras

## 🛠 Tech Stack

Python

TensorFlow / Keras

NumPy

Matplotlib

Streamlit

## 📊 Dataset

Brain MRI Dataset (Kaggle)

Classes:

glioma

meningioma

notumor

pituitary

Dataset not included due to size limitations.

## 🔍 Prediction Example

### Input MRI Image → pituitary (Confidence: 93.01%)
![Prediction Example](images/prediction.png)

## 🚀 How to Run

Clone the repository

Install dependencies

pip install -r requirements.txt

Run Streamlit app

streamlit run app.py

Or open the notebook:
BrainTumerDetection.ipynb

## 👩‍💻 Author

### Heba Shams
### AI & Backend Enthusiast 🤖✨
