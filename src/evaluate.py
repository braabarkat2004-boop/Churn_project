import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

from src.config import MODEL_PATH, THRESHOLD, REPORTS_DIR
from src.train import split_data


def evaluate_model():
    model = joblib.load(MODEL_PATH)
    x_train, x_test, y_train, y_test = split_data()

    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    print(f"Threshold used: {THRESHOLD}")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # save the plots
    REPORTS_DIR.mkdir(exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.savefig(REPORTS_DIR / "roc_curve.png", bbox_inches="tight")
    plt.close()

    return metrics


if __name__ == "__main__":
    evaluate_model()