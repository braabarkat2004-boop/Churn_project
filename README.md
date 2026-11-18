# Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to leave, with a REST API, a web dashboard, and a Docker image.

**Live demo:** <https://your-frontend-url.onrender.com>
**API docs:** <https://your-api-url.onrender.com/docs>

> The API runs on a free plan and sleeps after 15 minutes without traffic. The first prediction after a pause can take up to a minute.

![Dashboard screenshot](docs/screenshot.png)

## What it does

- Takes a customer profile (19 fields) and returns a churn probability and a Churn / No churn label.
- Shows the main patterns behind churn from the exploratory analysis.
- Suggests retention actions based on the customer's profile.

## Key findings

- 26.5% of the 7,043 customers churned.
- Contract type matters most: month-to-month customers churn at about 43%, one-year at 11%, two-year at 3%.
- Tenure has the strongest link to churn among numeric features (correlation −0.35).
- Customers without online security or tech support churn more.

## Model

- Logistic regression in a scikit-learn pipeline (Yeo-Johnson scaling for numeric columns, one-hot encoding for categories).
- Compared against Random Forest and XGBoost. Logistic regression gave the best balance of accuracy and recall at the same recall level.
- Decision threshold: `<0.xx>`, chosen to favor catching churners.

| Metric (test set) | Value |
|---|---|
| Recall | `<0.xx>` |
| Accuracy | `<0.xx>` |
| Precision | `<0.xx>` |

The threshold was chosen by comparing options on the test set, so these numbers are slightly optimistic.

## Project structure

```
├── Backend/        FastAPI app (Main.py, schemas.py)
├── frontend/       Dashboard (index.html)
├── src/            Pipeline: load, clean, features, train, evaluate, predict, registry
├── Models/         Trained model (churn_model.joblib)
├── notebooks/      Exploratory analysis (EDA.ipynb)
├── tests/          Data and model tests
├── Dockerfile
└── pyproject.toml
```

## Run it locally

```bash
# 1. install dependencies
uv sync

# 2. run the pipeline
python -m src.train
python -m src.evaluate
python -m src.registry      # optional: log the run to MLflow

# 3. start the API
python -m uvicorn Backend.Main:app --reload
```

Then open `frontend/index.html` in a browser. The API runs at `http://127.0.0.1:8000`, with docs at `/docs`.

## Run it with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Run the tests

```bash
python -m pytest -v
```

## API example

`POST /predict`

```json
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
  "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
}
```

Response:

```json
{ "churn_probability": 0.64, "churn_prediction": 1, "label": "Churn" }
```

## Tech stack

Python, pandas, scikit-learn, FastAPI, MLflow, Docker, pytest, HTML/CSS/JavaScript.
