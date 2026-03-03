# -----------------------------
# FORCE FILE-BASED MLFLOW (NO ALEMBIC)
# -----------------------------
import os
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

# -----------------------------
# Imports
# -----------------------------
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CreditCard_Fraud_Detection")

# -----------------------------
# Load data
# -----------------------------
DATA_DIR = "data/processed"
X_train = joblib.load(os.path.join(DATA_DIR, "X_train.pkl"))
X_test = joblib.load(os.path.join(DATA_DIR, "X_test.pkl"))
y_train = joblib.load(os.path.join(DATA_DIR, "y_train.pkl"))
y_test = joblib.load(os.path.join(DATA_DIR, "y_test.pkl"))

# -----------------------------
# Train & log model
# -----------------------------
with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=10,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    roc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("roc_auc", roc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Log model artifact (THIS CREATES model.pkl)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

    print("✅ Model training completed and logged to MLflow")
    print(f"ROC-AUC: {roc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
