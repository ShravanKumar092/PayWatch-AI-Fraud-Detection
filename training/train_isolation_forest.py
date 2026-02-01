import pandas as pd
from sklearn.ensemble import IsolationForest
from joblib import dump

X = pd.read_csv("data/X.csv")

iso = IsolationForest(
    n_estimators=300,
    contamination=0.01,
    random_state=42
)

iso.fit(X)

dump(iso, "api/models/isolation_forest.joblib")
