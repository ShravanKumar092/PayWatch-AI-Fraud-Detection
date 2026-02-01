import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.feature_engineering import feature_engineering

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Use the existing fraud dataset
df = pd.read_csv("Fraud_Analysis_Dataset.csv")

df = feature_engineering(df)

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

X.to_csv("data/X.csv", index=False)
y.to_csv("data/y.csv", index=False)
