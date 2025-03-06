import os
from flask import Flask, render_template, request
import joblib
import numpy as np
import json
import pandas as pd

app = Flask(__name__)

# Get absolute path of the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained model and scaler
model_path = os.path.join(BASE_DIR, "model", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")

# Check if files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Feature names for visualization
feature_names = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 
                'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate', 
                'Exercise Angina', 'ST Depression', 'ST Slope', 'Major Vessels', 'Thalassemia']

def get_risk_category(probability):
    """
    Categorize risk based on prediction probability
    """
    if probability < 0.2:
        return "Low Risk"
    elif probability < 0.5:
        return "Moderate Risk"
    elif probability < 0.8:
        return "High Risk"
    else:
        return "Very High Risk"

def get_recommendations(risk_category):
    """
    Provide recommendations based on risk category
    """
    recommendations = {
        "Low Risk": [
            "Maintain a healthy lifestyle",
            "Regular exercise",
            "Balanced diet",
            "Annual health checkups"
        ],
        "Moderate Risk": [
            "Increase physical activity",
            "Monitor blood pressure regularly",
            "Reduce salt intake",
            "Consider stress management techniques",
            "Schedule follow-up with healthcare provider"
        ],
        "High Risk": [
            "Immediate consultation with healthcare provider",
            "Daily blood pressure monitoring",
            "Strict diet control",
            "Regular exercise under medical supervision",
            "Stress reduction essential"
        ],
        "Very High Risk": [
            "Immediate medical attention required",
            "Follow prescribed medication regimen",
            "Frequent medical monitoring",
            "Lifestyle modifications under medical supervision",
            "Emergency plan in place"
        ]
    }
    return recommendations.get(risk_category, [])

def calculate_feature_contributions(input_data, model, scaler, feature_names):
    """
    Calculate feature contributions to the prediction
    """
    # Get the scaled input data
    scaled_input = scaler.transform(input_data)
    
    # Initialize contributions with zeros
    contributions = np.zeros(len(feature_names))
    
    # Get feature contributions (simplified approach)
    # For tree-based models (like RandomForest, XGBoost)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Calculate the contribution to this specific prediction
        contributions = importances * scaled_input[0]
        # Normalize to sum to 100%
        if np.sum(np.abs(contributions)) > 0:  # Avoid division by zero
            contributions = contributions / np.sum(np.abs(contributions)) * 100
    else:
        # If not a tree-based model, use a simple approach
        try:
            base_prediction = model.predict_proba(scaled_input)[0][1]
            for i in range(len(feature_names)):
                # Create a copy of the input with one feature zeroed
                modified_input = scaled_input.copy()
                modified_input[0, i] = 0
                modified_prediction = model.predict_proba(modified_input)[0][1]
                # The contribution is the difference in predictions
                contributions[i] = base_prediction - modified_prediction
            
            # Normalize to sum to 100%
            if np.sum(np.abs(contributions)) > 0:  # Avoid division by zero
                contributions = contributions / np.sum(np.abs(contributions)) * 100
        except:
            # If the model doesn't support predict_proba, assign equal weights
            contributions = np.ones(len(feature_names)) * (100.0 / len(feature_names))
    
    # Create feature contribution data
    feature_data = []
    for i, feature in enumerate(feature_names):
        # Convert numpy values to Python native types for JSON compatibility
        feature_data.append({
            'name': feature,
            'value': float(input_data[0][i]),
            'contribution': float(contributions[i]),
            'positive': bool(contributions[i] > 0)
        })
    
    # Sort by absolute contribution
    feature_data.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    return feature_data

def get_feature_value_description(feature_index, value):
    """
    Get human-readable description for feature values
    """
    feature_descriptions = {
        1: {0: "Female", 1: "Male"},  # Sex
        2: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"},  # CP
        5: {0: "≤ 120 mg/dl", 1: "> 120 mg/dl"},  # FBS
        6: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"},  # RestECG
        8: {0: "No", 1: "Yes"},  # ExAng
        10: {0: "Upsloping", 1: "Flat", 2: "Downsloping"},  # Slope
        12: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Not Available"},  # Thal
    }
    
    if feature_index in feature_descriptions and int(value) in feature_descriptions[feature_index]:
        return feature_descriptions[feature_index][int(value)]
    return str(value)

@app.route('/')
def welcome():
    # Return the page that will display the visuals
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract the form data from the POST request
        try:
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            cp = float(request.form['cp'])
            trestbps = float(request.form['trestbps'])
            chol = float(request.form['chol'])
            fbs = float(request.form['fbs'])
            restecg = float(request.form['restecg'])
            thalach = float(request.form['thalach'])
            exang = float(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = float(request.form['slope'])
            ca = float(request.form['ca'])
            thal = float(request.form['thal'])
        except ValueError:
            return render_template('predict.html', prediction="Invalid input. Please enter valid numbers.")

        # Prepare the input data as a numpy array and scale it
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        print(input_data)
        
        try:
            # Calculate feature contributions before scaling
            feature_data = calculate_feature_contributions(input_data, model, scaler, feature_names)
            
            # Add human-readable descriptions for categorical features
            for i, feature in enumerate(feature_data):
                feature_index = feature_names.index(feature['name'])
                if feature_index in [1, 2, 5, 6, 8, 10, 12]:  # Indices of categorical features
                    feature['description'] = get_feature_value_description(feature_index, feature['value'])
                else:
                    feature['description'] = str(feature['value'])
            
            # Scale the input data
            scaled_input = scaler.transform(input_data)
            
            # Predict using the model
            prediction = model.predict(scaled_input)[0]
            
            # Get prediction probability
            prediction_prob = model.predict_proba(scaled_input)[0][1]
            
            # Convert to native Python types for JSON compatibility
            prediction = int(prediction)
            prediction_prob = float(prediction_prob)
            
            # Get risk category and recommendations
            risk_category = get_risk_category(prediction_prob)
            recommendations = get_recommendations(risk_category)

            # Prepare result dictionary with serializable values
            result = {
                'prediction': "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
                'probability': round(prediction_prob * 100, 2),
                'risk_category': risk_category,
                'recommendations': recommendations,
                'feature_data': json.dumps(feature_data)  # Pre-serialize to avoid issues
            }

            return render_template('predict.html', result=result)
        except Exception as e:
            import traceback
            print(f"Error in prediction: {e}")
            print(traceback.format_exc())
            return render_template('predict.html', error=f"An error occurred during prediction: {str(e)}")

    return render_template('predict.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)