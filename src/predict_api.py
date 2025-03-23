from app import app

if __name__ == "__main__":
    print("Running the Flask app via predict_api.py")
    app.run(debug=True, port=5001)

from logging_config import configure_logging

# Get loggers for different modules
loggers = configure_logging()
logger = loggers['api']  # Get the logger specific to the 'api' module

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    try:
        data = request.get_json()
        logger.info(f"Received prediction request (v1) with data: {data}")
        
        # Prediction code
        result = model.predict([data])
        
        logger.info(f"Prediction successful: {result}")
        return jsonify({"prediction": result.tolist()})
    except Exception as e:
        logger.error(f"Prediction failed with error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting prediction API service")
    app.run(host="0.0.0.0", port=5000)