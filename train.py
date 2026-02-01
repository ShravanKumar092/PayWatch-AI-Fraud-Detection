import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import joblib
from xgboost import XGBClassifier

print("Starting model training process...")

# Define the feature engineering function (must be identical to the one in the app)
from src.feature_engineering import feature_engineering


# 1. Load the raw data
print("Loading dataset...")
raw_df = pd.read_csv('Fraud_Analysis_Dataset.csv')

# 2. Engineer features for the entire dataset
print("Engineering features...")
processed_df = feature_engineering(raw_df)

# 3. Prepare data for the model
X = processed_df.drop('isFraud', axis=1)
y = raw_df['isFraud'] # Use the original 'isFraud' column

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# 4. Train the definitive model
print("Training RandomForest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_f1 = f1_score(y_test, rf_preds)

print("Random Forest F1 Score:", rf_f1)

print("Model training complete.")

# 5. Save the trained model and the columns it expects
print("Saving model and columns to files...")
joblib.dump(rf_model, "rf_fraud_model.joblib")




xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)
xgb_f1 = f1_score(y_test, xgb_preds)

print("XGBoost F1 Score:", xgb_f1)
joblib.dump(xgb_model, "xgb_fraud_model.joblib")

# Save the final model (using XGBoost as it typically performs better)
joblib.dump(xgb_model, "fraud_model.joblib")
joblib.dump(X.columns, "model_columns.joblib")

print("\nProcess finished successfully!")
print("You can now run the Streamlit app: streamlit run app.py")