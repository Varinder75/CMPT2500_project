import pandas as pd
import yaml
import joblib

# Load configuration
with open("configs/predict_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract config values
model_path = config["model"]["model_load_path"]
input_data_path = config["data"]["input_data_path"]
output_predictions_path = config["data"]["output_predictions_path"]
features = config["features"]["selected"]

# Load the trained model
model = joblib.load(model_path)

# Load input data
data = pd.read_excel(input_data_path)  # Adjust based on file type (CSV, JSON, etc.)

# Select features for prediction
X = data[features]

# Make predictions
predictions = model.predict(X)

# Save predictions if enabled
if config["output"]["save_predictions"]:
    data["Predictions"] = predictions
    data.to_csv(output_predictions_path, index=False)
    print(f"Predictions saved to {output_predictions_path}")

print("Prediction completed successfully!")
