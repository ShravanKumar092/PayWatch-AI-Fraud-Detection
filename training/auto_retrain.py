import pandas as pd
import lightgbm as lgb
from joblib import dump
from src.feature_engineering import feature_engineering

df = pd.read_csv("data/retrain_data.csv")
df = feature_engineering(df)

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

model = lgb.LGBMClassifier(
    n_estimators=600,
    learning_rate=0.04,
    num_leaves=64,
    class_weight="balanced"
)

model.fit(X, y)

dump(model, "api/models/lightgbm_fraud.joblib")
dump(X.columns.tolist(), "api/models/model_columns.joblib")

print("✅ Model retrained and replaced safely")
