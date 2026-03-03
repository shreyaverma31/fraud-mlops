import joblib
import pandas as pd
import os

DATA_DIR = "data/processed"
OUT_DIR = "monitoring"

os.makedirs(OUT_DIR, exist_ok=True)

X_train = joblib.load(os.path.join(DATA_DIR, "X_train.pkl"))

df_ref = pd.DataFrame(X_train)
df_ref.to_csv(os.path.join(OUT_DIR, "reference.csv"), index=False)

print("✅ Reference dataset saved at monitoring/reference.csv")
