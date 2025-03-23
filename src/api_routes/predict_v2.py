from flask import Blueprint, request, jsonify

predict_v2 = Blueprint('predict_v2', __name__)

@predict_v2.route('/predict2', methods=['POST'])
def predict():
    try:
        # Example prediction, replace with actual model or logic for v2
        data = request.get_json()

        # Simulate a prediction result (Replace with actual logic)
        prediction = "Low" if data["Lag_1"] < 50 else "High"

        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
