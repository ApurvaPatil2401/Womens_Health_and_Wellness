from flask import Flask, request, jsonify , render_template
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS
from backend.preprocess import preprocess_data
import os
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Backend directory
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../models"))  # Model directory
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "../frontend"))  # Frontend directory
TEMPLATE_DIR = os.path.join(FRONTEND_DIR, "templates")  # HTML templates
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")  # CSS & JS

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app) 
try:
    gmm_model_path = os.path.join(MODEL_DIR, "pcos_gmm_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    # Check if files exist before loading
    if not os.path.exists(gmm_model_path):
        raise FileNotFoundError(f"Model file missing: {gmm_model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file missing: {scaler_path}")

    # Load the trained GMM model, PCA, and feature names
    with open(gmm_model_path, "rb") as file:
        loaded_data = pickle.load(file)

    print("Successfully loaded model file.")
    if isinstance(loaded_data, tuple) and len(loaded_data) == 3:
        gmm_model, pca, trained_feature_names = loaded_data
    else:
        raise ValueError(f"Unexpected format in pickle file: {loaded_data}")

    # Load the trained scaler
    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

except FileNotFoundError as e:
    print("Model file missing:", str(e))
    raise SystemExit("ERROR: Required model files are missing!")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        input_features = data["features"]
        if not isinstance(input_features, dict):
            return jsonify({"error": "Expected 'features' to be a dictionary with named values"}), 400

        # Convert input values to float
        try:
            input_numeric = {key: float(value) for key, value in input_features.items()}
        except ValueError:
            return jsonify({"error": "Invalid input: All values must be numbers"}), 400

        print("Received input:", input_numeric)

        # Preprocess input data
        input_processed = preprocess_data(input_data=input_numeric, train=False)
        input_processed["BMI*FastFood"] = input_processed["BMI"] * input_processed["Fast food (Y/N)"]
        input_processed["Exercise*HairLoss"] = input_processed["Reg.Exercise(Y/N)"] * input_processed["Hair loss(Y/N)"]

        if isinstance(input_processed, dict):
            input_processed = pd.DataFrame([input_processed])

        # Ensure all required columns are present
        missing_cols = [col for col in trained_feature_names if col not in input_processed.columns]
        if missing_cols:
            return jsonify({"error": f"Missing columns in input: {missing_cols}"}), 400

        input_processed = input_processed[trained_feature_names]
        input_processed = input_processed.to_numpy().reshape(1, -1)

        # Apply feature scaling
        input_scaled = scaler.transform(input_processed)  
        input_pca = pca.transform(input_scaled)

        # Predict cluster with probabilities
        cluster_probs = gmm_model.predict_proba(input_pca)[0]
        cluster = np.argmax(cluster_probs)
        
        # Risk levels & recommendations (Reassigned dynamically)
        cluster_means = np.mean(gmm_model.means_, axis=1)
        sorted_clusters = np.argsort(cluster_means)
        risk_levels = {sorted_clusters[0]: "High Risk", sorted_clusters[1]: "Moderate Risk", sorted_clusters[2]: "Low Risk"}
        recommendations = {
            sorted_clusters[2]: [
                "Continue with a healthy lifestyle and regular check-ups.",
                "Maintain a balanced diet with whole foods, lean protein, and healthy fats.",
                "Continue regular physical activity (at least 3 times per week).",
                "Keep hydrated and get 7-9 hours of sleep daily.",
                "Monitor menstrual health & visit a doctor annually for checkups."
            ],
            sorted_clusters[1]: [
                "Maintain a balanced diet and monitor symptoms regularly.",
                "Improve diet: Reduce sugar & processed foods.",
                "Exercise: Yoga, pilates, or brisk walking (30 min daily).",
                "Manage stress with meditation, deep breathing, or therapy.",
                "Track ovulation & menstrual cycles.",
                "Consider natural supplements (Spearmint tea for hormone balance)."
            ],
            sorted_clusters[0]: [
                "Consult a doctor and follow a strict diet and exercise plan",
                "Consult a gynecologist for medical evaluation and treatment.",
                "Follow a low-carb, high-fiber diet (whole grains, legumes, greens).",
                "Engage in regular exercise (30 min daily - cardio & strength training).",
                "Consider supplements (Vitamin D, Myo-Inositol, Omega-3s).",
                "Monitor menstrual cycles & hormonal changes closely."
            ]
        }

        return jsonify({
            "cluster": int(cluster),
            "risk": risk_levels.get(cluster, "Unknown Risk"),
            "recommendation": recommendations.get(cluster, ["No recommendations available"])
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
