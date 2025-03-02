from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from backend.clustering import predict_cluster
from flask_cors import CORS

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets the backend directory
MODEL_PATH = os.path.join(BASE_DIR, "../models/pcos_gmm_model.pkl")  # Moves one level up

# Load pre-trained model
model, feature_names = pickle.load(open(MODEL_PATH, "rb"))

app = Flask(__name__)
CORS(app)  # Add this to enable cross-origin requests

@app.route("/")
def home():
    return "Welcome to PCOS Health Advisor API"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Received Data:", data)  # Debugging print

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        input_features = np.array(data["features"]).reshape(1, -1)

        # Convert input to DataFrame with correct column names
        #feature_names = ["age", "weight", "height", "bmi", "exercise"]
        input_df = pd.DataFrame(input_features, columns=feature_names)

        # Predict cluster and recommendation
        cluster, recommendation = predict_cluster(input_df, model)

        return jsonify({"cluster": int(cluster), "recommendation": recommendation})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
     app.run(debug=True, host="0.0.0.0", port=5000)
