import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from joblib import dump

X = pd.read_csv("data/X.csv")
y = pd.read_csv("data/y.csv").values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    class_weight="balanced"
)

model.fit(X_train, y_train)

dump(model, "api/models/lightgbm_fraud.joblib")
dump(X.columns.tolist(), "api/models/model_columns.joblib")
