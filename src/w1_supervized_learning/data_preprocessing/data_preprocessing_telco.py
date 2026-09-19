# This preprocessing is for the Telco dataset. 

import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv

load_dotenv()

path_dataset_telco = os.getenv("PATH_DATASET_TELCO")
telco_churn_df = pd.read_csv(path_dataset_telco)

# Not PaperBilling, PaymentMethod,

# Remove features
columns = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
           "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
           "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
           "Contract", "MonthlyCharges", "TotalCharges", "Churn"]
telco_churn_df = telco_churn_df[columns]


# Encode gender
telco_churn_df["gender"] = telco_churn_df["gender"].map({"Male": 0, "Female": 1})

# Encode Partner
telco_churn_df["Partner"] = telco_churn_df["Partner"].map({"No": 0, "Yes": 1})

# Encode Dependents
telco_churn_df["Dependents"] = telco_churn_df["Dependents"].map({"No": 0, "Yes": 1})

# Encode PhoneService
telco_churn_df["PhoneService"] = telco_churn_df["PhoneService"].map({"No": 0, "Yes": 1})

# Encode MultipleLines
telco_churn_df["MultipleLines"] = telco_churn_df["MultipleLines"].map({"No": 0, "Yes": 1, "No phone service":2})

# Encode InternetService
telco_churn_df["InternetService"] = telco_churn_df["InternetService"].map({"DSL": 0, "Fiber optic": 1, "No":2})

# Encode OnlineSecurity
telco_churn_df["OnlineSecurity"] = telco_churn_df["OnlineSecurity"].map({"No": 0, "Yes": 1, "No internet service":2})

# Encode OnlineBackup
telco_churn_df["OnlineBackup"] = telco_churn_df["OnlineBackup"].map({"No": 0, "Yes": 1, "No internet service":2})

# Encode DeviceProtection
telco_churn_df["DeviceProtection"] = telco_churn_df["DeviceProtection"].map({"No": 0, "Yes": 1, "No internet service":2})

# Encode TechSupport
telco_churn_df["TechSupport"] = telco_churn_df["TechSupport"].map({"No": 0, "Yes": 1, "No internet service":2})

# Encode StreamingTV
telco_churn_df["StreamingTV"] = telco_churn_df["StreamingTV"].map({"No": 0, "Yes": 1, "No internet service":2})

# Encode StreamingMovies
telco_churn_df["StreamingMovies"] = telco_churn_df["StreamingMovies"].map({"No": 0, "Yes": 1, "No internet service":2})

# Encode Contract
telco_churn_df["Contract"] = telco_churn_df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year":2})

# Encode Churn
telco_churn_df["Churn"] = telco_churn_df["Churn"].map({"No": 0, "Yes": 1})

print("Datos sin borrar NaN:", len(telco_churn_df))

# Drop NaN values

# TotalCharges has some attributes in blank
# Clean TotalCharges
telco_churn_df["TotalCharges"] = pd.to_numeric(
    telco_churn_df["TotalCharges"], errors="coerce"
)

telco_churn_df = telco_churn_df.dropna()

print("Datos borrados los  NaN:", len(telco_churn_df))


print(telco_churn_df.head())


# Display the first few rows of the dataset
#print(telco_churn_df.head())

# Split datasets
X = telco_churn_df.drop("Churn", axis=1)
y = telco_churn_df["Churn"]


# Scale values
scaler_telco_churn = StandardScaler()
X = scaler_telco_churn.fit_transform(X)

#print(X)

X_telco_churn, X_telco_churn, y_telco_churn, y_telco_churn = train_test_split(
    X, y, test_size=0.2, random_state=42
)