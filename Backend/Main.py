import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Backend.schemas import Customer
from src.predict import predict

app = FastAPI(title="Customer Churn API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    result = predict(df).to_dict(orient="records")[0]

    if result["churn_prediction"] == 1:
        result["label"] = "Churn"
    else:
        result["label"] = "No churn"

    return result