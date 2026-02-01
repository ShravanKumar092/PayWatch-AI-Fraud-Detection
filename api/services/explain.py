import os
import shap
import pandas as pd
import joblib

# Compute models directory relative to this file to avoid cwd issues
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "lightgbm_fraud.joblib"))
columns = joblib.load(os.path.join(MODEL_DIR, "model_columns.joblib"))

explainer = shap.TreeExplainer(model)

def explain(tx):
    df = pd.DataFrame([tx]).reindex(columns=columns, fill_value=0)
    values = explainer.shap_values(df)[1][0]
    return dict(zip(columns, values))
