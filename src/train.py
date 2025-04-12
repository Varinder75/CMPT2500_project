# src/train.py
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import time  # For measuring training duration
from src.preprocess import load_and_clean_data

# Import monitoring module
from utils.monitoring import TrainingMonitor

# Initialize Monitoring (Prometheus server will start on port 8000)
monitor = TrainingMonitor(port=8000)

# Ensure MLflow is connected to the correct tracking URI
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)


def train_model(data_path, n_estimators=100, max_depth=None):
    with mlflow.start_run(run_name=f"rf_n_estimators_{n_estimators}_max_depth_{max_depth}") as run:
        try:
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            
            data = load_and_clean_data(data_path)
            if data is None:
                print("Error: Data loading failed.")
                return

            print("Data loaded successfully.")
            print(data.head())

            features = ['Lag_1', 'Lag_2', 'Lag_3', 'Number of employees']
            target = 'Release_Category'

            X = data[features]
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            start_time = time.time()

            # Monitor Epoch (we only have 1 epoch for RF)
            monitor.epoch_gauge.set(1)

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            end_time = time.time()
            duration = end_time - start_time
            print(f"Training completed in {duration:.2f} seconds.")

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Validation Accuracy: {accuracy}")

            # Log to MLflow
            mlflow.log_metric("validation_accuracy", accuracy)

            # Monitor Accuracy
            monitor.val_accuracy_gauge.set(accuracy)

            # Save model
            model_output_path = "rf_model.pkl"
            joblib.dump(model, model_output_path)
            mlflow.log_artifact(model_output_path)

        except Exception as e:
            print(f"An error occurred during training: {e}")


if __name__ == "__main__":
    data_path = "data/your_dataset.csv"
    train_model(data_path)
