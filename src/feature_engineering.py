import pandas as pd
import numpy as np

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Centralized, production-safe feature engineering
    Used for:
    - training
    - inference
    - API
    """

    df = df.copy()

    # ----------------------------------
    # DROP NON-INFORMATIVE STRING COLUMNS
    # ----------------------------------
    df = df.drop(columns=['nameOrig', 'nameDest'], errors='ignore')

    # ----------------------------------
    # ENSURE REQUIRED NUMERIC COLUMNS EXIST
    # ----------------------------------
    required_numeric_cols = [
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'step'
    ]

    for col in required_numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # ----------------------------------
    # ONE-HOT ENCODE TRANSACTION TYPE
    # ----------------------------------
    if 'type' in df.columns:
        df['type'] = df['type'].astype(str)
        df = pd.get_dummies(
            df,
            columns=['type'],
            prefix='type',
            drop_first=False
        )
    else:
        # Ensure at least one dummy column exists for safety
        df['type_UNKNOWN'] = 1

    # ----------------------------------
    # FRAUD-SPECIFIC ENGINEERED FEATURES
    # ----------------------------------
    df['error_balance_orig'] = (
        df['newbalanceOrig'] - df['oldbalanceOrg'] + df['amount']
    )

    df['error_balance_dest'] = (
        df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']
    )

    # ----------------------------------
    # CYCLICAL TIME FEATURES (SAFE)
    # ----------------------------------
    df['step_sin'] = np.sin(2 * np.pi * df['step'] / 24)
    df['step_cos'] = np.cos(2 * np.pi * df['step'] / 24)

    return df
