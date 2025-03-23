import mlflow.sklearn
import numpy as np
import pandas as pd

class ModelPredictor:
    def __init__(self, run_id):
        """
        Initialize the predictor with an MLflow run ID.
        :param run_id: str, the MLflow run ID where the model is logged.
        """
        model_uri = f"runs:/{run_id}/model"
        self.model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully from MLflow.")

    def predict(self, input_data: pd.DataFrame):
        """
        Make a prediction using the loaded model.
        :param input_data: pd.DataFrame containing the features for prediction.
        :return: np.ndarray of model predictions.
        """
        predictions = self.model.predict(input_data)
        return predictions

if __name__ == "__main__":
    # Replace <RUN_ID> with the actual run ID from MLflow
    predictor = ModelPredictor(run_id="b1e5ee3a1af04b03bee0a8f389da52ad")

    # Define sample input data
    sample_input = np.array([[20, 14, 48, 100]])  # Example input
    features = ['Lag_1', 'Lag_2', 'Lag_3', 'Number of employees']
    
    # Convert to DataFrame with correct column names
    sample_input_df = pd.DataFrame(sample_input, columns=features)
    
    # Get predictions
    preds = predictor.predict(sample_input_df)
    print("Predictions:", preds)

from logging_config import configure_logging

# Get loggers for different modules
loggers = configure_logging()
logger = loggers['predict']  # Get the logger specific to the 'predict' module

def predict(data):
    try:
        logger.info(f"Received prediction request with data: {data}")
        result = model.predict(data)
        logger.info(f"Prediction successful: {result}")
        return result
    except Exception as e:
        logger.error(f"Prediction failed with error: {str(e)}")
        raise