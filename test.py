# import pandas as pd
# from src.predict import predict

# customer = {
#     "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
#     "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
#     "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
#     "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
#     "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
#     "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
# }
# print(predict(pd.DataFrame([customer])))

from src.load_data import load_dataset
from src.clean_data import clean_data

df = clean_data(load_dataset())
nulls = df.isnull().sum()
print(nulls[nulls > 0])