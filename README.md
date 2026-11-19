# Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to leave. It includes a data pipeline, a trained model, a REST API (FastAPI, Docker), and a web dashboard.

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/baraabarkat/churnapp)
**API docs:** [Swagger UI](https://churn-project-e8e18.containers.snapdeploy.app/docs)

> The API runs on a free plan and goes to sleep after about 15 minutes without traffic. The first prediction after a pause can take up to a minute. If the dashboard says it can't reach the API, open the [health check](https://churn-project-e8e18.containers.snapdeploy.app/health), wait for the response, and try again.

![Dashboard screenshot](docs/screenshot.png)

## What it does

- Takes a customer profile (19 fields) and returns a churn probability and a Churn / No churn label.
- Shows the main patterns behind churn from the exploratory analysis.
- Suggests retention actions based on the customer's profile.

## Key findings

- 26.5% of the 7,043 customers churned.
- Contract type matters most: month-to-month customers churn at about 43%, one-year at 11%, two-year at 3%.
- Tenure has the strongest link to churn among numeric features (correlation -0.35).
- Customers without online security or tech support churn more.
- Higher monthly bills are linked to more churn, but the relationship is not linear, and it is mixed up with tenure, contract type and fiber optic service.

## Model

- Logistic regression inside a scikit-learn pipeline: Yeo-Johnson scaling for numeric columns, one-hot encoding for categorical ones.
- I compared it with Random Forest and XGBoost. At the same recall level, logistic regression gave the best accuracy and precision, so the extra complexity did not help.
- The decision threshold is `<0.xxxx>`. It is set below 0.5 on purpose, to catch more churners at the cost of more false alarms.

| Metric (test set) | Value |
|---|---|
| Recall | `<0.xx>` |
| Accuracy | `<0.xx>` |
| Precision | `<0.xx>` |

**Limitation:** the threshold was chosen by comparing options on the test set, so these numbers are slightly optimistic. A cross-validated tuning step would give more reliable estimates.

## Architecture

```
Browser (Hugging Face static Space)  ->  FastAPI (Docker, SnapDeploy)  ->  scikit-learn pipeline
```

- **Dashboard:** a single HTML page that sends the customer data to the API and shows the result.
- **API:** FastAPI with input validation (Pydantic). Endpoints: `GET /health` and `POST /predict`.
- **Model:** the full pipeline (preprocessing and classifier) is saved with joblib, so training and prediction use identical steps.

## Project structure

```
├── Backend/        FastAPI app (Main.py, schemas.py)
├── frontend/       Dashboard (index.html)
├── src/            Pipeline code
│   ├── config.py       paths, columns, threshold, model parameters
│   ├── load_data.py    read and check the raw data
│   ├── clean_data.py   cleaning shared by training and the API
│   ├── features.py     preprocessing
│   ├── train.py        split, train, save the model
│   ├── evaluate.py     metrics and plots
│   ├── predict.py      load the model and score customers
│   └── registry.py     log runs to MLflow
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

The API runs at `http://127.0.0.1:8000`, with interactive docs at `/docs`.
To use the dashboard with your local API, set the `API` line in `frontend/index.html` to `http://127.0.0.1:8000` and open the file in a browser.

## Run it with Docker

```bash
docker build -t churn-api .
docker run --rm -p 8000:8000 churn-api
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

Requests with invalid values (for example a lowercase `"female"` or a negative tenure) are rejected with a `422` error.

## Tech stack

Python, pandas, scikit-learn, FastAPI, Pydantic, MLflow, Docker, pytest, HTML/CSS/JavaScript.

## Data

IBM Telco Customer Churn dataset (7,043 customers, 20 features).
