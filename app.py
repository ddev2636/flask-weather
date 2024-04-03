from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from joblib import load

app = Flask(__name__)
CORS(app)

# Load the model during Flask app initialization
model = load("models/your_model.joblib")

# Define the expected feature names
EXPECTED_FEATURES = [
    "jan",
    "feb",
    "march",
    "april",
    "may",
    "june",
    "july",
    "aug",
    "sept",
]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = request.json

        # Validate feature presence and numerical values
        for key in EXPECTED_FEATURES:
            if key not in features:
                return jsonify({"error": f"Missing feature: {key}"}), 400
            try:
                # Attempt to convert value to float
                features[key] = float(features[key])
            except ValueError:
                return (
                    jsonify({"error": f"Non-numerical value for feature: {key}"}),
                    400,
                )

        # Normalize the input features using the loaded scaler
        input_data = np.array([[features[key] for key in EXPECTED_FEATURES]])

        # Make prediction
        prediction = model.predict(input_data)

        # Return the prediction
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        print(f"Error making prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(port=5000)
