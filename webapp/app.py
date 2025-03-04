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
        input_data = scaler.transform(input_data)

        # Predict using the model
        prediction = model.predict(input_data)[0]

        # Return the prediction result
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        return render_template('predict.html', prediction=result)

    return render_template('predict.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
