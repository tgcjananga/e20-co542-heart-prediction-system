import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class HeartDiseaseANN:
    def __init__(self):
        """
        Initializes the ANN model and the scaler.
        """
        self.scaler = StandardScaler()
        self.model = None

    def load_data(self, file_path):
        """
        Loads and preprocesses the dataset.
        """
        data = pd.read_csv(file_path)
        X = data.drop(columns=["target"])  # Features
        y = data["target"]  # Labels

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def build_model(self):
        """
        Builds the ANN model with the specified architecture.
        """
        self.model = Sequential([
            Dense(32, activation='relu', input_shape=(13,)),  # Input layer
            Dense(16, activation='relu'),  # Hidden layer 1
            Dense(8, activation='relu'),   # Hidden layer 2
            Dense(1, activation='sigmoid') # Output layer for binary classification
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, epochs=50, batch_size=16):
        """
        Trains the ANN model.
        """
        if not self.model:
            self.build_model()
        
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    def evaluate(self):
        """
        Evaluates the trained ANN model.
        """
        y_pred = self.model.predict(self.X_test)
        y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary (0 or 1)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

    def save_model(self, model_path="model/ann_model.h5", scaler_path="model/scaler.pkl"):
        """
        Saves the trained ANN model and scaler.
        """
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print("Model and scaler saved successfully.")

# ==========================
# Run the ANN model
# ==========================

if __name__ == "__main__":
    file_path = "data/heart_disease.csv"

    ann_model = HeartDiseaseANN()
    ann_model.load_data(file_path)
    ann_model.train()
    ann_model.evaluate()
    ann_model.save_model("model/ann_model.h5", "model/scaler.pkl")
