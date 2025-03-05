import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import argparse
import os

# Load dataset
data = pd.read_csv("data/heart_disease.csv")

# Define input and output features
X = data.drop(columns=["target"])  # 'target' is the output column
y = data["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train ANN Model
model = MLPClassifier(hidden_layer_sizes=(32, 16, 8), activation='relu', max_iter=500)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save Model and Scaler
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
