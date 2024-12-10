import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# Initialize Flask
app = Flask(__name__)

# Path for data files
CSV_FILE = "/app/food-data.csv"
LABELS_FILE = "/app/labels.json"
DATA_FILE = "data_store.json"

# Initialize data store
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w') as f:
        json.dump([], f)

with open(DATA_FILE, 'r') as f:
    data_store = json.load(f)

# Read CSV
try:
    df = pd.read_csv(CSV_FILE)
    print("Dataset successfully loaded.")
except FileNotFoundError:
    print(f"Error: File {CSV_FILE} not found.")
    df = pd.DataFrame()  # Create empty DataFrame to prevent errors


# Load model
try:
    model = tf.keras.models.load_model('/app/model_nutrition_stat.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def validate_input_data(tb, bb, usia, jenis_kelamin):
    """Validate input data with additional checks."""
    if tb < 50 or tb > 220:
        return False, "Invalid height (50-220 cm)"
    if bb < 10 or bb > 250:
        return False, "Invalid weight (10-250 kg)"
    if usia < 1 or usia > 100:
        return False, "Invalid age (1-100 years)"
    bmi = bb / ((tb/100)**2)
    if usia < 18:
        if bmi < 10 or bmi > 35:
            return False, f"Unrealistic BMI for age {usia} (BMI: {bmi:.2f})"
    else:
        if bmi < 15 or bmi > 40:
            return False, f"Unrealistic adult BMI (BMI: {bmi:.2f})"
    if jenis_kelamin not in ["Laki-laki", "Perempuan"]:
        return False, "Gender must be 'Laki-laki' or 'Perempuan'"
    return True, ""

def get_food_recommendations(category, n=5):
    """Get food recommendations based on nutritional category"""
    try:
        if category == "Gizi Baik":
            filtered_data = df[(df["Caloric Value"] > 50) & (df["Caloric Value"] < 200) & (df["Protein"] > 5)]
        elif category == "Gizi Kurang":
            filtered_data = df[(df["Caloric Value"] >= 200) & (df["Protein"] > 10)]
        elif category == "Gizi Lebih":
            filtered_data = df[(df["Caloric Value"] < 50) & (df["Fat"] < 5)]
        else:
            return []

        if filtered_data.empty:
            return []
        return filtered_data.sample(min(n, len(filtered_data)))[["food", "Caloric Value", "Protein"]].to_dict(orient="records")
    except Exception as e:
        print(f"Error in food recommendations: {e}")
        return []

@app.route("/", methods=["GET"])
def index():
    """Basic landing page"""
    return "Welcome to NutriKids Nutrition Prediction Service!"

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data"}), 400

        # Input validation
        required_keys = ["tb", "bb", "usia", "jenis_kelamin"]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing key: {key}"}), 400

        tb, bb, usia, jenis_kelamin = data["tb"], data["bb"], data["usia"], data["jenis_kelamin"]

        # Validate input data
        is_valid, error_message = validate_input_data(tb, bb, usia, jenis_kelamin)
        if not is_valid:
            return jsonify({"error": error_message}), 400

        # Calculate BMI
        bmi = bb / ((tb/100)**2)

        # Determine nutrition status based on BMI
        labels = ["Gizi Baik", "Gizi Kurang", "Gizi Lebih"]
        if bmi < 17:
            predicted_index = 1
        elif 17 <= bmi < 25:
            predicted_index = 0
        else:
            predicted_index = 2

        prediction_label = labels[predicted_index]

        # Get food recommendations
        recommendations = get_food_recommendations(prediction_label, n=5)

        # Define descriptions
        descriptions = {
            "Gizi Baik": "Child has appropriate weight for age and height.",
            "Gizi Kurang": "Child has lower weight than recommended.",
            "Gizi Lebih": "Child has higher weight than recommended."
        }

        # Prepare response
        response_data = {
            "bmi": round(bmi, 2),
            "prediction": prediction_label,
            "description": descriptions[prediction_label],
            "recommendations": recommendations,
        }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Prediction processing failed", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)