import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Load the model during initialization
def init():
    global model
    model_path = Model.get_model_path('heart-disease-model')  # Ensure this model name matches your Azure ML registration
    model = joblib.load(model_path)

# Define the scoring route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame(data['data'])
        predictions = model.predict(input_data)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    init()
    app.run(host='0.0.0.0', port=5001)
