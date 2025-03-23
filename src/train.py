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
from preprocess import load_and_clean_data

# Ensure MLflow is connected to the correct tracking URI
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

def train_model(data_path, n_estimators=100, max_depth=None):
    with mlflow.start_run(run_name=f"rf_n_estimators_{n_estimators}_max_depth_{max_depth}") as run:
        try:
            # Log parameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            
            # Load and preprocess data
            data = load_and_clean_data(data_path)
            if data is None:
                print("Error: Data loading failed.")
                return

            print("Data loaded successfully.")
            print(data.head())  # Debug: check first few rows of data

            features = ['Lag_1', 'Lag_2', 'Lag_3', 'Number of employees']
            target = 'Release_Category'

            X = data[features]
            y = data[target]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Start timing the model training
            start_time = time.time()

            # Initialize and train the model with dynamic hyperparameters
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            training_duration = time.time() - start_time

            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model Accuracy: {accuracy * 100:.2f}%")

            # Log metrics (accuracy)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("training_duration", training_duration)

            # Log the model to MLflow
            print(f"Logging model to MLflow...")
            mlflow.sklearn.log_model(model, "random_forest_model")
            print(f"Model logged to MLflow with run ID: {mlflow.active_run().info.run_id}")

            # Optionally, log the model locally (for backup)
            joblib.dump(model, 'models/my_model.pkl')
            print("Model saved locally to 'models/my_model.pkl'")

            mlflow.log_param("model_training_duration_seconds", training_duration)

        except Exception as e:
            print(f"Error during training: {e}")
            mlflow.log_param("error", str(e))


if __name__ == "__main__":
    # Experiment 1
    train_model('data/raw/mydata.xlsx', n_estimators=200, max_depth=10)

    # Experiment 2
    train_model('data/raw/mydata.xlsx', n_estimators=150, max_depth=5)

    # Experiment 3
    train_model('data/raw/mydata.xlsx', n_estimators=100, max_depth=None)

from logging_config import configure_logging

# Get loggers for different modules
loggers = configure_logging()
logger = loggers['train']  # Get the logger specific to the 'train' module

def train_model():
    logger.info("Starting model training")
    # Your training code here
    accuracy = 0.85  # Example accuracy
    logger.info(f"Model trained successfully with accuracy: {accuracy}")