from flask import Blueprint, jsonify

health_status = Blueprint('health', __name__)

@health_status.route('/status', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200
