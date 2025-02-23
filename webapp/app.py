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

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(request.form[key]) for key in request.form]
    
    # Scale input data
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction="Heart Disease Detected" if prediction == 1 else "No Heart Disease")

if __name__ == '__main__':
    app.run(debug=True)
