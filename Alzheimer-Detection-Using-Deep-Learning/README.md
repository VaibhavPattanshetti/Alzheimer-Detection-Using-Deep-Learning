# 🧠 Alzheimer’s Disease Detection from MRI using Deep Learning

A deep learning–based medical imaging system to classify **Alzheimer’s Disease vs Normal brain MRI scans** using **ResNet50V2 transfer learning** with high recall and AUC.

---

## 📌 Project Overview

This project builds an end-to-end deep learning pipeline to automatically detect Alzheimer’s Disease from MRI images.  
The goal is to assist early diagnosis by developing a reliable, high-performance classification model.

---

## 🧠 Problem Statement

Early diagnosis of Alzheimer’s Disease is critical but challenging.  
Manual MRI interpretation is time-consuming and depends heavily on expert availability.

This project applies **deep learning and transfer learning** to automate MRI classification and support clinical decision-making.

---

## 🗂 Dataset

**Source:** Kaggle – MRI Brain Scans (Neurological Disorders Dataset)

**Classes used:**
- Alzheimer’s Disease  
- Normal  

**Preprocessing steps:**
- Invalid/corrupted image removal  
- Channel consistency checks  
- Image resizing to 224×224  
- Data augmentation  

---

## ⚙️ Methodology

- Exploratory Data Analysis (EDA) on MRI datasets  
- Data cleaning and preprocessing  
- Transfer Learning using **ResNet50V2 (ImageNet weights)**  
- Two-phase training strategy:
  - Phase 1: Feature extraction (base layers frozen)  
  - Phase 2: Fine-tuning deeper layers  
- Class imbalance handling using class weights  
- Evaluation using Accuracy, AUC, Recall, Confusion Matrix, ROC Curve  

---

## 🏗 Model Architecture

- ResNet50V2 (pretrained)  
- Global Average Pooling  
- Batch Normalization  
- Dense + Dropout layers  
- Sigmoid output for binary classification  

---

## 📊 Results

| Metric | Value |
|-------|--------|
Validation Accuracy | **94.58%**  
Validation AUC | **0.9914**  
Validation Recall | **0.9182**

High recall was prioritized to minimize **false negatives**, which is critical in medical diagnosis.

---

## 📈 Visual Results

- Training vs Validation Accuracy & Loss  
- Confusion Matrix  
- ROC Curve  

(Available inside the `/results` folder)

---

## 🚀 Key Learnings

- Medical image preprocessing techniques  
- Transfer learning and fine-tuning strategies  
- Evaluation-driven ML model development  
- Handling class imbalance in healthcare datasets  
- Building end-to-end deep learning pipelines  

---

## ▶ How to Run

Install dependencies:

```bash
pip install -r requirements.txt
