import joblib
import pandas as pd

from src.config import MODEL_PATH, THRESHOLD, TARGET
from src.clean_data import clean_data
_model = None   # loaded once then reused
def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model
def predict(input_df):
    df = clean_data(input_df)
    df = df.drop(columns=[TARGET], errors="ignore")

    model = load_model()
    prob = model.predict_proba(df)[:, 1]

    result = pd.DataFrame({
        "churn_probability": prob.round(4),
        "churn_prediction": (prob >= THRESHOLD).astype(int),
    })
    return result