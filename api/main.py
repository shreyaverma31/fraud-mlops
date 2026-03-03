from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib
import numpy as np
import os
import logging

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Credit Card Fraud Detection API")

# -----------------------------
# Ensure logs folder exists
# -----------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

# -----------------------------
# Load artifacts
# -----------------------------
SCALER_PATH = "data/processed/scaler.pkl"
MODEL_PATH = "models/model/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ model.pkl not found")

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

# -----------------------------
# Request schema
# -----------------------------
class Transaction(BaseModel):
    features: conlist(float, min_length=30, max_length=30)

# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(txn: Transaction):
    try:
        X = np.array(txn.features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prob = model.predict_proba(X_scaled)[0][1]
        prediction = int(prob >= 0.5)

        # 🔥 Monitoring log
        logging.info(
            f"prediction={prediction}, probability={prob:.4f}, features_mean={np.mean(txn.features):.4f}"
        )

        return {
            "prediction": prediction,
            "probability": round(float(prob), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
