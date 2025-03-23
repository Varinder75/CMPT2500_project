from flask import Flask, jsonify
from api_routes.predict_v1 import predict_v1
from api_routes.predict_v2 import predict_v2
from api_routes.home import ml_project_home  # Import the home blueprint

app = Flask(__name__)

# Register the blueprints with proper URL prefixes
app.register_blueprint(ml_project_home, url_prefix='/v1')  # Register /home endpoint
app.register_blueprint(predict_v1, url_prefix='/v1')  # Register v1 predict routes
app.register_blueprint(predict_v2, url_prefix='/v2')  # Register v2 predict routes

@app.route('/')
def home_route():
    return "Welcome to the Flask API!"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True, port=5001)
