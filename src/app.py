from flask import Flask, jsonify
from api_routes.predict_v1 import predict_v1
from api_routes.predict_v2 import predict_v2
from api_routes.home import ml_project_home
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import generate_latest, REGISTRY

from prometheus_client import Counter, Histogram, Gauge
import time
import psutil
import threading
import os

app = Flask(__name__)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Custom metrics
prediction_requests = Counter('model_prediction_requests_total', 'Total number of prediction requests', ['model_version', 'status'])
prediction_time = Histogram('model_prediction_duration_seconds', 'Time spent processing prediction', ['model_version'])
memory_usage = Gauge('app_memory_usage_bytes', 'Memory usage of the application')
cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage percentage of the application')

# Register the blueprints with proper URL prefixes
app.register_blueprint(ml_project_home, url_prefix='/v1')
app.register_blueprint(predict_v1, url_prefix='/v1')
app.register_blueprint(predict_v2, url_prefix='/v2')

@app.route('/')
def home_route():
    return "Welcome to the Flask API!"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Explicitly ensure the Prometheus metrics route is set up
@app.route('/metrics')
def metrics_route():
    from prometheus_client import generate_latest, REGISTRY
    return generate_latest(REGISTRY)

# Add a function to monitor resource usage in the background
def monitor_resources():
    """Update system resource metrics every 15 seconds"""
    while True:
        process = psutil.Process(os.getpid())
        memory_usage.set(process.memory_info().rss)  # in bytes
        cpu_usage.set(process.cpu_percent())
        time.sleep(15)

# Start resource monitoring in a background thread when app starts
if __name__ == "__main__":
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    
    # Run the Flask app
    print("Starting Flask app...")
    print(app.url_map)
    app.run(debug=True, port=5002)
