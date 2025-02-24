import joblib
import json
import numpy as np
import os
from azureml.core.model import Model
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model on startup
model_path = Model.get_model_path("heart-disease-model")
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()
        input_data = np.array(data["input"]).reshape(1, -1)  # Ensure correct shape

        # Make predictions
        prediction = model.predict(input_data).tolist()

        # Return response
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
