from flask import Flask, jsonify
from data_store import window_data

app = Flask(__name__)

@app.route('/get_window_data', methods=['GET'])
def get_window_data():
    global window_data  # Use the global variable
    return jsonify(window_data)
