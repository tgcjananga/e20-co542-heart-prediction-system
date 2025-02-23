# **Heart Disease Prediction using Artificial Neural Networks (ANN)**  

## **Project Overview**  
This project focuses on developing an **Artificial Neural Network (ANN)** model to predict **heart disease** based on key health parameters. Heart disease is a leading cause of death worldwide, and early detection can significantly improve patient outcomes. By leveraging **machine learning and deep learning techniques**, this system aims to provide an **accurate and efficient** prediction model for healthcare professionals.

## **Team Information**  
**Project Title:** Heart Disease Prediction using ANN  
**Group Name:** CoreMind  
**Team Members:**  
- E/20/453  
- E/20/158  
- E/20/300  
- E/20/248  
- E/20/377  

---

## **Problem Statement**  
Traditional heart disease diagnosis requires a **series of expensive and time-consuming** medical tests. Our goal is to create an ANN-based model that can analyze **patient health data** and predict heart disease **with high accuracy**, assisting doctors in making informed decisions.

## **Motivation**  
- **Traditional methods** require multiple medical tests, increasing cost and time.  
- **Machine Learning (ML) models**, especially ANNs, can detect complex patterns in patient data.  
- A **neural network-based approach** has the potential to improve accuracy and enable early diagnosis.  

---

## **Project Scope & Objectives**  
- Develop an **ANN-based model** for heart disease classification.  
- Utilize a dataset containing key health indicators like **blood pressure, cholesterol, heart rate, and ECG results**.  
- Perform **data preprocessing** (handling missing values, encoding categorical variables, normalizing continuous variables).  
- Train and evaluate the ANN model using **performance metrics** such as accuracy, precision, recall, and F1-score.  
- Deploy the model and integrate **MLOps tools** for tracking and lifecycle management.  

---

## **Dataset Information**  
We will use the **UCI Heart Disease Dataset**, which contains **13 key features** related to heart health:  

### **Input Features (13 Variables)**  
- Age  
- Sex  
- Chest pain type (4 categories)  
- Resting blood pressure (mm Hg)  
- Serum cholesterol (mg/dl)  
- Fasting blood sugar (>120 mg/dl)  
- Resting ECG results (normal, ST-T wave abnormality, etc.)  
- Maximum heart rate achieved  
- Exercise-induced angina  
- Old peak (ST depression induced by exercise)  
- Slope of the peak exercise ST segment  
- Number of major vessels (0–3) colored by fluoroscopy  
- Thalassemia (normal, fixed defect, reversible defect)  

### **Output Variable**  
- **Binary classification:**  
  - `0` = No heart disease  
  - `1` = Presence of heart disease  

---

## **Neural Network Architecture**  
Our proposed ANN model consists of:  
- **Input Layer**: 13 neurons (one per feature)  
- **Hidden Layers**:  
  - 32 neurons, ReLU activation  
  - 16 neurons, ReLU activation  
  - 8 neurons, ReLU activation  
- **Output Layer**: 1 neuron, Sigmoid activation (binary classification)  

### **Data Preprocessing Steps**
- Handling missing values (imputation or removal).  
- Encoding categorical variables (one-hot encoding).  
- Normalizing continuous variables using **MinMaxScaler** or **StandardScaler**.  
- Splitting the dataset into **80% training and 20% testing**.  

### **Model Training & Evaluation**  
- **Loss Function**: Binary Cross-Entropy  
- **Optimizer**: Adam Optimizer  
- **Batch Size**: 32  
- **Epochs**: 50–100  
- **Performance Metrics**:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
  - AUC-ROC Curve  
  - Confusion Matrix  

---

## **MLOps Integration and Deployment**  
- **Experiment Tracking**: Use **MLFlow** for logging experiments, hyperparameter tuning, and performance tracking.  
- **Model Deployment**: Implement a **web-based interface** using **Flask** or **Streamlit** to accept user input and provide predictions.  
- **Monitoring & Management**: Automate **model versioning and performance monitoring** in real-time.  

---

## **Expected Deliverables**  
1. **Trained ANN Model**: Predicts heart disease with high accuracy.  
2. **Evaluation Report**: Performance metrics, visualization, and feature analysis.  
3. **User Interface**: Simple web-based front end for input and predictions.  
4. **Final Presentation**: Summary of findings, model performance, and deployment results.  

---

## **Project Setup & Installation**  
To run this project on your local system, follow these steps:

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/YOUR-ORG/YOUR-REPO.git
cd YOUR-REPO
