import joblib
import mlflow
import mlflow.sklearn

from src.config import (
    MODEL_PATH, BEST_PARAMS, THRESHOLD,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,
)
from src.evaluate import evaluate_model


def log_to_mlflow():
    metrics = evaluate_model()
    model = joblib.load(MODEL_PATH)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="Customer_Churn") as run:
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="final_model")

    print("Run ID:", run.info.run_id)


if __name__ == "__main__":
    log_to_mlflow()