import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from preprocess import load_and_clean_data

def evaluate_model(run_id, data_path):
    # Load the model from MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print("Model loaded successfully from MLflow.")

    # Load and preprocess the data
    data = load_and_clean_data(data_path)
    
    if data is None:
        print("Data could not be loaded or cleaned. Exiting.")
        return
    else:
        print("Data loaded and cleaned successfully.")
        print(f"Data columns: {data.columns}")
        print(f"First few rows of data:\n{data.head()}")

    # Prepare data for evaluation
    features = ['Lag_1', 'Lag_2', 'Lag_3', 'Number of employees']
    target = 'Release_Category'

    if not all(col in data.columns for col in features + [target]):
        print("Error: Required columns are missing in the data.")
        return

    X = data[features]
    y = data[target]

    # Predictions
    y_pred = model.predict(X)

    # Evaluation metrics
    print("\nClassification Report:\n", classification_report(y, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:\n", cm)

if __name__ == "__main__":
    # Replace <RUN_ID> with the actual run ID from MLflow
    evaluate_model('b1e5ee3a1af04b03bee0a8f389da52ad', '../data/raw/mydata.xlsx')
