import os
import joblib
import pandas as pd
from api.services.behavior_rules import velocity_rule, drift_rule

# Compute models directory relative to this file to avoid cwd issues
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

lgb_model = joblib.load(os.path.join(MODEL_DIR, "lightgbm_fraud.joblib"))
iso_model = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.joblib"))
columns = joblib.load(os.path.join(MODEL_DIR, "model_columns.joblib"))

def fraud_decision(tx, tx_count=0, avg_amount=0):
    df = pd.DataFrame([tx]).reindex(columns=columns, fill_value=0)

    lgb_prob = lgb_model.predict_proba(df)[0][1]
    iso_score = iso_model.decision_function(df)[0]

    risk = "LOW"
    if lgb_prob > 0.85 or iso_score < -0.15:
        risk = "HIGH"
    elif lgb_prob > 0.6:
        risk = "MEDIUM"

    if velocity_rule(tx_count) or drift_rule(tx["amount"], avg_amount):
        risk = "HIGH"

    return {
        "fraud_probability": float(lgb_prob),
        "anomaly_score": float(iso_score),
        "risk_level": risk
    }
