# api_routes/home.py

from flask import Blueprint, jsonify

# Create a blueprint instance
ml_project_home = Blueprint('ml_project_home', __name__)

# Define the home route
@ml_project_home.route('/home', methods=['GET'])
def home():
    response = {
        "message": "Welcome to the Prediction API!",
        "description": "This API predicts the classification based on the input features.",
        "valid_request_payload": {
            "Lag_1": "Numeric value for Lag 1",
            "Lag_2": "Numeric value for Lag 2",
            "Lag_3": "Numeric value for Lag 3",
            "Number of employees": "Numeric value representing the number of employees"
        },
        "example_request": {
            "Lag_1": 10,
            "Lag_2": 20,
            "Lag_3": 30,
            "Number of employees": 100
        },
        "available_endpoints": {
            "/health": "Check the health status of the API",
            "/v1/predict1": "Predict the outcome based on provided features"
        }
    }
    return jsonify(response), 200
