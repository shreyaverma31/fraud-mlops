import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Paths
DATA_PATH = "data/creditcard.csv"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_data():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save processed data
    joblib.dump(X_train, f"{OUTPUT_DIR}/X_train.pkl")
    joblib.dump(X_test, f"{OUTPUT_DIR}/X_test.pkl")
    joblib.dump(y_train, f"{OUTPUT_DIR}/y_train.pkl")
    joblib.dump(y_test, f"{OUTPUT_DIR}/y_test.pkl")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

    print("✅ Data preprocessing completed and saved.")

if __name__ == "__main__":
    preprocess_data()
