import pandas as pd 
from src.config import RAW_DATA_PATH , TARGET,ID_COL,BINARY_COLS,NUM_COLS,PASSTHROUGH_COLS,CAT_COLS

def load_dataset():
    df=pd.read_csv(RAW_DATA_PATH)
    expected_cols=[TARGET,ID_COL]+BINARY_COLS+NUM_COLS+PASSTHROUGH_COLS+CAT_COLS
    missing=[]
    for col in expected_cols:
        if col not in df.columns:
            missing.append(col)
    if len(missing)>0:
        raise ValueError(f"Missing columns in the data: {missing}")

    return df 