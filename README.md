# 🚀 MobileNetV2 Transfer Learning — Cats vs Dogs Classification

A Deep Learning project demonstrating how to **modify and fine-tune a pre-trained CNN** for a new image classification task using **TensorFlow & Keras**.

---

## 📌 Project Overview

This project implements **Transfer Learning and Fine-Tuning** using the pre-trained **MobileNetV2** architecture (trained on ImageNet) to solve a binary image classification problem:

> 🐱 Cat vs 🐶 Dog Classification

Instead of training a CNN from scratch, we:

1. Load a pre-trained convolutional base  
2. Remove the original classification head  
3. Add a new custom classifier  
4. Train the new head (Feature Extraction stage)  
5. Fine-tune upper layers with a lower learning rate  

This approach significantly reduces training time and improves performance when working with limited datasets.

---

## 🧠 Key Concepts Covered

- Transfer Learning  
- Fine-Tuning  
- Feature Extraction  
- Data Augmentation  
- Binary Image Classification  
- Pre-trained CNN Models  
- TensorFlow Datasets (TFDS)  

---

## 📂 Dataset

Dataset used: **`cats_vs_dogs`** from TensorFlow Datasets.

- ~23,000 labeled images  
- 2 classes: `cat` and `dog`  
- 80% training — 20% validation split  
- Images resized to `224×224`  

### Data Augmentation Techniques
- Random horizontal flip  
- Random rotation  
- Random zoom  

---

## 🏗 Model Architecture

### 🔹 Base Model
- MobileNetV2  
- Pre-trained on ImageNet  
- `include_top=False`  
- Frozen during initial training  

### 🔹 Custom Classification Head
- GlobalAveragePooling2D  
- Dropout (0.2)  
- Dense (1 neuron, Sigmoid activation)  

---

## 🔄 Training Strategy

### Stage 1 — Feature Extraction
- Base model frozen  
- Train only new classifier  
- Optimizer: Adam  
- Learning rate: `1e-3`

### Stage 2 — Fine-Tuning
- Unfreeze top layers of MobileNetV2  
- Lower learning rate (`1e-5`)  
- Continue training to adapt high-level features  

---

## 📊 Results

- High validation accuracy achieved using transfer learning  
- Fine-tuning improved performance further  
- Demonstrates efficiency of pre-trained models for real-world applications  

> Note: Final accuracy may vary depending on hardware and training epochs.

---

## 🛠 Tech Stack

- Python  
- TensorFlow 2.x  
- Keras  
- TensorFlow Datasets  
- Matplotlib  

---

## ▶️ How to Run

### Option 1 — Google Colab (Recommended)

1. Upload `final_assignment.ipynb`
2. Enable GPU  
   `Runtime → Change runtime type → GPU`
3. Run all cells

### Option 2 — Local Environment

```bash
pip install tensorflow tensorflow-datasets matplotlib
