import os
from flask import Flask, render_template, request
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

def get_risk_category(probability):
    if probability < 0.2:
        return "Low Risk"
    elif probability < 0.5:
        return "Moderate Risk"
    elif probability < 0.8:
        return "High Risk"
    else:
        return "Very High Risk"

def get_recommendations(risk_category):
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

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Create a list of feature names in the correct order
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        # Get input values in the correct order
        input_data = [float(request.form[feature]) for feature in feature_names]
        
        # Debug print
        print("Input data before scaling:", input_data)
        
        # Scale input data
        input_data = np.array(input_data).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)
        
        # Debug print
        print("Input data after scaling:", input_data_scaled)

        # Get prediction probability
        prediction_prob = model.predict_proba(input_data_scaled)[0][1]
        prediction = model.predict(input_data_scaled)[0]
        
        # Debug print
        print("Prediction:", prediction)
        print("Prediction probability:", prediction_prob)
        
        # Get risk category and recommendations
        risk_category = get_risk_category(prediction_prob)
        recommendations = get_recommendations(risk_category)

        result = {
            'prediction': "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
            'probability': round(prediction_prob * 100, 2),
            'risk_category': risk_category,
            'recommendations': recommendations
        }

        return render_template('index.html', result=result)
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")  # Debug print
        return render_template('index.html', error=f"Error in prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True) 