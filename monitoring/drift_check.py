import pandas as pd
from scipy.stats import ks_2samp

print("Checking for data drift...")

# Load original training data
train_data = pd.read_csv("data/creditcard.csv")
train_sample = train_data.drop("Class", axis=1).sample(1000, random_state=42)

# Load new prediction inputs from logs (mock example)
# In real system you'd parse logged data
# Here we simulate new data sample
new_sample = train_sample.sample(1000, random_state=24)

drift_detected = False

for column in train_sample.columns:
    stat, p_value = ks_2samp(train_sample[column], new_sample[column])
    
    if p_value < 0.05:
        print(f"Drift detected in column: {column}")
        drift_detected = True

if not drift_detected:
    print("No significant data drift detected.")
