from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = xgb.XGBRegressor()
model.load_model("xgb_env_model_temporal.json")

print("✅ Model loaded successfully")

# -----------------------------
# HEALTH CHECK (optional)
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return "Coral Bleaching API is running"

# -----------------------------
# PREDICTION ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "input" not in data:
            return jsonify({"error": "Missing input"}), 400

        input_data = data["input"]

        # Ensure correct length
        if len(input_data) != 11:
            return jsonify({
                "error": f"Expected 11 features, got {len(input_data)}"
            }), 400

        # Convert to numpy
        arr = np.array(input_data).reshape(1, -1)

        # Predict
        prediction = model.predict(arr)[0]

        return jsonify({
            "prediction": float(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
