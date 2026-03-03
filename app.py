from flask import Flask, request, jsonify
import joblib
import numpy as np
import datetime

app = Flask(__name__)

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Fraud Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]

    input_data = np.array(data).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    # Logging prediction
    with open("monitoring/logs/predictions.log", "a") as f:
        f.write(f"{datetime.datetime.now()} | Input: {data} | Prediction: {prediction}\n")

    return jsonify({
        "prediction": int(prediction),
        "message": "Fraud Detected!" if prediction == 1 else "Legitimate Transaction"
    })

if __name__ == "__main__":
    app.run(debug=True)

