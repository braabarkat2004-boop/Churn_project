import pandas as pd
from src.config import ID_COL, TARGET, BINARY_COLS


def clean_data(df):
    df_clean = df.copy()
    df_clean = df_clean.drop(columns=[ID_COL], errors="ignore")
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median())
    for col in BINARY_COLS:
        df_clean[col] = df_clean[col].map({"Yes": 1, "No": 0})
    df_clean["gender"] = df_clean["gender"].map({"Male": 1, "Female": 0})
    if TARGET in df_clean.columns:
        df_clean[TARGET] = df_clean[TARGET].map({"Yes": 1, "No": 0})

    return df_clean