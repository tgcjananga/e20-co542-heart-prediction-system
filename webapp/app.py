import os
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

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

# Feature descriptions for better explanation
FEATURE_NAMES = [
    "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Cholesterol",
    "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", "Exercise-Induced Angina",
    "ST Depression", "Slope of ST Segment", "Major Vessels Count", "Thalassemia"
]

@app.route('/')
def home():
    return render_template('index.html', prediction=None, explanation=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[key]) for key in request.form]
        input_data = np.array(input_data).reshape(1, -1)
        input_data = scaler.transform(input_data)

        # Predict probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]  # Probability of heart disease
            percentage_risk = round(probability * 100, 2)
            prediction_text = f"You have a {percentage_risk}% risk of heart disease."
        else:
            prediction = model.predict(input_data)[0]
            prediction_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        # Feature importance (if the model supports it)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_contributions = dict(zip(FEATURE_NAMES, importances))
            sorted_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
        else:
            sorted_features = None

        # Health Recommendations
        health_tips = ""  # Default empty
        if percentage_risk >= 70:
            health_tips = "High risk detected. Consult a cardiologist immediately and adopt a heart-healthy lifestyle."
        elif 40 <= percentage_risk < 70:
            health_tips = "Moderate risk. Consider dietary improvements, regular exercise, and monitoring your condition."
        elif 20 <= percentage_risk < 40:
            health_tips = "Low to moderate risk. Maintain a balanced diet, stay active, and check up regularly."
        else:
            health_tips = "Minimal risk detected. Keep up with a healthy lifestyle to prevent future risks."

        return render_template('index.html', prediction=prediction_text, explanation=sorted_features, health_tips=health_tips)
    except ValueError:
        return render_template('index.html', prediction="Invalid input. Please enter numeric values.")

if __name__ == '__main__':
    app.run(debug=True)