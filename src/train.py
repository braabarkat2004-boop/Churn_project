import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import TARGET, RANDOM_STATE, TEST_SIZE, MODEL_DIR, MODEL_PATH, BEST_PARAMS
from src.load_data import load_dataset
from src.clean_data import clean_data
from src.features import build_preprocessor


def split_data():
    """Load, clean and split the data. evaluate.py reuses this function,
    so both scripts get exactly the same train/test split."""
    df = clean_data(load_dataset())

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test


def build_model():
    """Full pipeline: preprocessor + classifier with the tuned parameters."""
    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            **BEST_PARAMS,
            random_state=RANDOM_STATE,
        )),
    ])
    return model


def train():
    x_train, x_test, y_train, y_test = split_data()

    model = build_model()
    model.fit(x_train, y_train)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()