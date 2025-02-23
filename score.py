import json
import joblib
import numpy as np
import os
import torch
from azureml.core.model import Model
from sklearn.preprocessing import StandardScaler

# Load the model during initialization
def init():
    global model, scaler

    # Get model path
    model_path = Model.get_model_path('heart-disease-model')
    scaler_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'scaler.pkl')

    # Load the model
    model = joblib.load(model_path)

    # Load the scaler (if used for preprocessing)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = None

# Run function for inference
def run(raw_data):
    try:
        # Parse input JSON
        data = json.loads(raw_data)
        
        # Convert input to NumPy array
        input_data = np.array(data['data'])

        # Preprocess data if scaler exists
        if scaler:
            input_data = scaler.transform(input_data)

        # Make prediction
        predictions = model.predict(input_data)

        # Convert to list for JSON response
        return json.dumps({'predictions': predictions.tolist()})

    except Exception as e:
        return json.dumps({'error': str(e)})
