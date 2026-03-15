# app1.py
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# --- Global variable to hold latest ESP data ---
latest_sensor_data = {}
MAIN_BACKEND_URL = "http://127.0.0.1:5000/sensor_predict_from_esp"  # main Flask app endpoint

@app.route('/test', methods=['POST'])
def receive_from_esp():
    global latest_sensor_data
    data = request.get_json()
    if data:
        latest_sensor_data = data
        print("✅ Data received from ESP8266:", latest_sensor_data)

        # Forward to main Flask app for prediction
        try:
            response = requests.post(MAIN_BACKEND_URL, json=data)
            print("📤 Forwarded to main app:", response.status_code, response.text)
        except Exception as e:
            print("❌ Error sending to main app:", e)

        return jsonify({"status": "success", "received": latest_sensor_data})
    else:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400


@app.route('/get_latest_data', methods=['GET'])
def get_latest_data():
    global latest_sensor_data
    if latest_sensor_data:
        return jsonify(latest_sensor_data)
    else:
        return jsonify({"status": "error", "message": "No data yet"}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
