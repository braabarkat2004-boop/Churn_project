from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from src.config import NUM_COLS, PASSTHROUGH_COLS, CAT_COLS


def build_preprocessor():
    num_pipe = Pipeline([
        ("power", PowerTransformer(method="yeo-johnson"))
    ])
    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("binary", "passthrough", PASSTHROUGH_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])
    return preprocessor