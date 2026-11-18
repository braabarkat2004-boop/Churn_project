from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = PROJECT_DIR / "Models"          # matches your existing folder name
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
REPORTS_DIR = PROJECT_DIR / "reports"
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD= 0.29378663616695644
TARGET = "Churn"
ID_COL = "customerID"
BINARY_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
PASSTHROUGH_COLS = ["gender", "SeniorCitizen"] + BINARY_COLS
BEST_PARAMS = {
 'penalty': 'l1',
 'solver': 'saga',
 'C': 0.7076278814637156,
 'max_iter': 1391}

CAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_DIR / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT = "Customer_Churn"