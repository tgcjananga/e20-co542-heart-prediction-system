# **Heart Disease Prediction using Machine Learning**

---

## **Project Overview**

This project focuses on developing a **Machine Learning model** to predict **heart disease** based on key health parameters. Heart disease is a leading cause of death worldwide, and early detection can significantly improve patient outcomes. By leveraging **machine learning and MLOps techniques**, this system aims to provide an **accurate and efficient** prediction model for healthcare professionals.

### 🎯 **Key Features**

- **Real-time Prediction**: Instant heart disease risk assessment
- **98% Accuracy**: XGBoost classifier trained on UCI Heart Disease Dataset
- **Risk Categorization**: Low, Moderate, High, and Very High risk levels
- **Feature Analysis**: Visual breakdown of contributing factors
- **Personalized Recommendations**: Health advice based on risk level
- **Interactive Dashboard**: User-friendly web interface with visualizations
- **MLOps Integration**: Automated CI/CD pipeline with Azure ML and GitHub Actions

---

## **Team Information**

**Project Title:** Heart Disease Prediction System  
**Group Name:** CoreMind  
**Team Members:**
- E/20/453
- E/20/158
- E/20/300
- E/20/248
- E/20/377

**Course**: CO542 - Machine Learning  
**Institution**: University of Peradeniya  
**Department**: Computer Engineering  
**Year**: 2024/2025

---

## **Problem Statement**

Traditional heart disease diagnosis requires a **series of expensive and time-consuming** medical tests. Our goal is to create a machine learning model that can analyze **patient health data** and predict heart disease **with high accuracy**, assisting doctors in making informed decisions.

## **Motivation**

- **Traditional methods** require multiple medical tests, increasing cost and time
- **Machine Learning models** can detect complex patterns in patient data
- **ML-based approach** has the potential to improve accuracy and enable early diagnosis
- **Automated deployment** ensures rapid integration into healthcare systems

---

## **Project Scope & Objectives**

- Develop a **machine learning model** for heart disease classification
- Utilize a dataset containing key health indicators like **blood pressure, cholesterol, heart rate, and ECG results**
- Perform **data preprocessing** (handling missing values, encoding categorical variables, normalizing continuous variables)
- Train and evaluate the model using **performance metrics** such as accuracy, precision, recall, and F1-score
- Deploy the model and integrate **MLOps tools** for tracking and lifecycle management
- Implement **CI/CD pipelines** for automated training and deployment

---

## **Dataset Information**

We use the **UCI Heart Disease Dataset**, which contains **13 key features** related to heart health.

### **Input Features (13 Variables)**

| Feature | Description | Type |
|---------|-------------|------|
| Age | Age in years | Numeric |
| Sex | Gender (0=Female, 1=Male) | Categorical |
| CP | Chest pain type (0-3) | Categorical |
| Trestbps | Resting blood pressure (mm Hg) | Numeric |
| Chol | Serum cholesterol (mg/dl) | Numeric |
| FBS | Fasting blood sugar > 120 mg/dl | Binary |
| RestECG | Resting ECG results (0-2) | Categorical |
| Thalach | Maximum heart rate achieved | Numeric |
| Exang | Exercise induced angina | Binary |
| Oldpeak | ST depression induced by exercise | Numeric |
| Slope | Slope of peak exercise ST segment | Categorical |
| CA | Number of major vessels (0-3) | Numeric |
| Thal | Thalassemia (0-3) | Categorical |

### **Output Variable**

- **Binary classification:**
  - `0` = No heart disease
  - `1` = Presence of heart disease

---

## **Machine Learning Model**

### **Model Architecture**

We implemented an **XGBoost Classifier** with the following configuration:
- **Algorithm**: Gradient Boosting Decision Trees
- **Number of Estimators**: 100
- **Learning Rate**: 0.1
- **Max Depth**: 6
- **Random State**: 42 (for reproducibility)

### **Why XGBoost?**
- Excellent performance on tabular data
- Built-in feature importance
- Robust to outliers
- Fast training and inference
- High interpretability

### **Data Preprocessing Steps**

1. **Duplicate Removal**: Remove duplicate entries from dataset
2. **Feature Scaling**: Normalize continuous variables using **StandardScaler**
3. **Train-Test Split**: Split dataset into **80% training and 20% testing**
4. **Feature Engineering**: Extract relevant patterns from raw data

### **Model Training & Evaluation**

- **Training Method**: Supervised Learning with Cross-Validation
- **Optimization**: Gradient Boosting with Early Stopping
- **Performance Metrics**:
  - **Accuracy**: 98.54%
  - **Precision**: High
  - **Recall**: High
  - **F1-Score**: Balanced
  - **AUC-ROC**: 0.989
  - **Confusion Matrix**: Detailed prediction analysis

---

## **MLOps Integration and Deployment**

### **CI/CD Pipeline**

Automated workflows using **GitHub Actions**:

#### 1. **Setup Workflow** (`setup.yml`)
- Creates Azure ML Workspace
- Configures resource groups
- Initializes cloud infrastructure

#### 2. **Training Workflow** (`train_model.yml`)
- Uploads training data to Azure ML
- Submits training jobs
- Tracks experiments and metrics

#### 3. **Deployment Workflow** (`deploy_model.yml`)
- Registers trained model
- Creates/updates online endpoints
- Deploys model as REST API
- Handles endpoint failures automatically
- Reduces deployment time by 80%

### **Infrastructure as Code**

- **Azure ML Workspace Configuration**: YAML-based setup
- **Compute Clusters**: Automated provisioning
- **Managed Online Endpoints**: Scalable inference service
- **Environment Management**: Containerized dependencies

### **Model Deployment**

- **Framework**: Flask web application
- **Inference**: Real-time predictions via REST API
- **Monitoring**: Performance tracking and logging
- **Versioning**: Model registry for version control

---

## **Web Application**

### **Home Page**
- Dataset overview with statistics
- Interactive visualizations:
  - Correlation heatmap
  - Target distribution
  - Gender analysis
  - Disease prevalence by demographics

### **Prediction Page**
- 13-field input form with validation
- Real-time risk assessment
- Visual risk probability meter
- Feature contribution chart (Chart.js)
- Personalized health recommendations

### **Risk Categories**

| Category | Probability Range | Recommendations |
|----------|------------------|-----------------|
| **Low Risk** | 0-20% | Maintain healthy lifestyle, annual checkups |
| **Moderate Risk** | 20-50% | Increase physical activity, monitor BP |
| **High Risk** | 50-80% | Immediate consultation, strict diet control |
| **Very High Risk** | 80-100% | Immediate medical attention required |

---

## **Project Setup & Installation**

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Git

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/tgcjananga/e20-co542-heart-prediction-system
cd e20-co542-heart-prediction-system
```

### **2️⃣ Create Virtual Environment**
```bash
python -m venv venv
```

### **3️⃣ Activate Virtual Environment**

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### **4️⃣ Install Dependencies**
```bash
pip install Flask joblib numpy pandas scikit-learn xgboost matplotlib seaborn
```

### **5️⃣ Train the Model** (Optional - pre-trained model included)
```bash
python train.py
```

### **6️⃣ Test the Model** (Optional - evaluate performance)
```bash
python test.py
```

### **7️⃣ Run the Web Application**
```bash
python webapp/app.py
```

### **8️⃣ Access the Application**
Open your browser and navigate to: **`http://localhost:5000`**

---

## **Project Structure**

```
e20-co542-heart-prediction-system/
├── .github/
│   └── workflows/              # CI/CD pipelines
│       ├── setup.yml          # Azure workspace setup
│       ├── train_model.yml    # Automated training
│       └── deploy_model.yml   # Automated deployment
├── azureml/                    # Azure ML configurations
│   ├── train.yml              # Training job config
│   ├── deploy.yml             # Deployment config
│   └── environment.yml        # Dependencies
├── data/
│   └── heart_disease.csv      # UCI dataset
├── model/
│   ├── xgboost_model.pkl      # Trained model
│   └── scaler.pkl             # Feature scaler
├── results/                    # Test results & metrics
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── model_metrics.csv
├── webapp/
│   ├── app.py                 # Flask application
│   ├── analysis.py            # Data visualization
│   ├── static/                # CSS, images
│   │   └── visuals/          # Generated charts
│   └── templates/             # HTML templates
│       ├── index.html
│       └── predict.html
├── train.py                    # Model training script
├── test.py                     # Model testing script
├── score.py                    # Azure ML scoring script
├── requirements.txt            # Dependencies
├── .gitignore
└── README.md
```

---

## **Technology Stack**

### **Core Technologies**
- **Python 3.8+**: Programming language
- **XGBoost 2.0.0**: Machine learning algorithm
- **scikit-learn**: ML utilities and preprocessing
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### **Web Framework**
- **Flask 2.3.0**: Web application framework
- **Chart.js**: Interactive visualizations
- **HTML/CSS**: Frontend design

### **Data Visualization**
- **matplotlib**: Static plots
- **seaborn**: Statistical visualization

### **DevOps & Cloud**
- **GitHub Actions**: CI/CD automation
- **Azure ML**: Cloud ML platform
- **YAML**: Configuration management
- **joblib**: Model serialization

---

## **Performance Metrics**

```
╔══════════════════════════════════════╗
║     MODEL PERFORMANCE METRICS        ║
╠══════════════════════════════════════╣
║  Accuracy:         98.54%            ║
║  Precision:        97.8%             ║
║  Recall:           98.2%             ║
║  F1-Score:         98.0%             ║
║  ROC-AUC:          0.9891            ║
╚══════════════════════════════════════╝
```

---

## **Expected Deliverables**

✅ **Trained ML Model**: XGBoost model predicting heart disease with 98% accuracy  
✅ **Evaluation Report**: Comprehensive performance metrics and visualizations  
✅ **Web Application**: Interactive Flask-based user interface  
✅ **CI/CD Pipeline**: Automated training and deployment workflows  
✅ **Documentation**: Complete project documentation and setup guide  
✅ **Test Results**: Model evaluation with confusion matrix and ROC curves

---

## **Testing**

### **Run Model Tests**
```bash
python test.py
```

### **Generated Test Outputs**
- Confusion matrix visualization
- ROC curve analysis
- Precision-recall curve
- Feature importance chart
- Performance metrics CSV
- Classification report

All test results are saved in the `results/` directory.

---

## 🔮 **Future Enhancements**

- [ ] Implement deep learning models (ANN with TensorFlow/Keras)
- [ ] Add SHAP values for advanced explainability
- [ ] Create mobile application (React Native/Flutter)
- [ ] Integrate with Electronic Health Records (EHR)
- [ ] Add multi-language support
- [ ] Implement patient history tracking
- [ ] Real-time model monitoring dashboard
- [ ] A/B testing framework for model versions
- [ ] Ensemble methods combining multiple models

---

## **Architecture Diagram**

```
┌─────────────────────────────────────────────────────┐
│              User Interface (Flask)                  │
│    Input Form | Risk Dashboard | Visualizations     │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│           Application Layer                          │
│  • Input Validation                                  │
│  • Feature Engineering                               │
│  • Risk Assessment                                   │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│          ML Model (XGBoost)                          │
│  • Trained on UCI Dataset                            │
│  • StandardScaler Normalization                      │
│  • Feature Importance Analysis                       │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│    CI/CD Pipeline (GitHub Actions + Azure ML)        │
│  • Automated Training                                │
│  • Model Registry                                    │
│  • Deployment Automation                             │
└─────────────────────────────────────────────────────┘
```

---

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 **Contributing**

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 **Contact**

For questions or feedback, please contact the team through the [GitHub repository](https://github.com/tgcjananga/e20-co542-heart-prediction-system).

---

## 🙏 **Acknowledgments**

- **UCI Machine Learning Repository** for the heart disease dataset
- **University of Peradeniya, Department of Computer Engineering**
- Our project supervisor for guidance and support
- Open-source community for amazing tools and libraries

---


Made with ❤️ by Team CoreMind

</div>
