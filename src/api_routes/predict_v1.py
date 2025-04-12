from flask import Blueprint, request, jsonify
import joblib
import os

predict_v1 = Blueprint('predict_v1', __name__)

# Dynamically create absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '../../models/my_model.pkl')

# Load model once at the start
model_v1 = joblib.load(model_path)

@predict_v1.route('/predict1', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required_fields = ["Lag_1", "Lag_2", "Lag_3", "Number of employees"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        input_data = [[data['Lag_1'], data['Lag_2'], data['Lag_3'], data['Number of employees']]]

        prediction = model_v1.predict(input_data)

        return jsonify({"prediction": prediction[0]}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
