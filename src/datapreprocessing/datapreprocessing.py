import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv

load_dotenv()

path_dataset_telco = os.getenv("PATH_DATASET_TELCO")
customer_churn_df = pd.read_csv(path_dataset_telco)

# Not PaperBilling, PaymentMethod,

# Remove features
columns = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
           "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
           "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
           "Contract", "MonthlyCharges", "TotalCharges", "Churn"]
customer_churn_df = customer_churn_df[columns]


# Encode gender
customer_churn_df["gender"] = customer_churn_df["gender"].map({"Male": 0, "Female": 1})

# Encode Partner
customer_churn_df["Partner"] = customer_churn_df["Partner"].map({"No": 0, "Yes": 1})

# Encode Dependents
customer_churn_df["Dependents"] = customer_churn_df["Dependents"].map({"No": 0, "Yes": 1})

# Encode PhoneService
customer_churn_df["PhoneService"] = customer_churn_df["PhoneService"].map({"No": 0, "Yes": 1})

# Encode MultipleLines
customer_churn_df["MultipleLines"] = customer_churn_df["MultipleLines"].map({"No": 0, "Yes": 1, "No phone service":2})

# Encode InternetService
customer_churn_df["InternetService"] = customer_churn_df["InternetService"].map({"DSL": 0, "Fiber optic": 1, "No":2})

# Encode OnlineSecurity
customer_churn_df["OnlineSecurity"] = customer_churn_df["OnlineSecurity"].map({"No": 0, "Yes": 1, "No internet service":2})

# Encode OnlineBackup
customer_churn_df["OnlineBackup"] = customer_churn_df["OnlineBackup"].map({"No": 0, "Yes": 1, "No internet service":2})

# Encode DeviceProtection
customer_churn_df["DeviceProtection"] = customer_churn_df["DeviceProtection"].map({"No": 0, "Yes": 1, "No internet service":2})

# Encode TechSupport
customer_churn_df["TechSupport"] = customer_churn_df["TechSupport"].map({"No": 0, "Yes": 1, "No internet service":2})

# Encode StreamingTV
customer_churn_df["StreamingTV"] = customer_churn_df["StreamingTV"].map({"No": 0, "Yes": 1, "No internet service":2})

# Encode StreamingMovies
customer_churn_df["StreamingMovies"] = customer_churn_df["StreamingMovies"].map({"No": 0, "Yes": 1, "No internet service":2})

# Encode Contract
customer_churn_df["Contract"] = customer_churn_df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year":2})

# Encode Churn
customer_churn_df["Churn"] = customer_churn_df["Churn"].map({"No": 0, "Yes": 1})

print("Datos sin borrar NaN:", len(customer_churn_df))

# Drop NaN values

# TotalCharges has some attributes in blank
# Clean TotalCharges
customer_churn_df["TotalCharges"] = pd.to_numeric(
    customer_churn_df["TotalCharges"], errors="coerce"
)

customer_churn_df = customer_churn_df.dropna()

print("Datos borrados los  NaN:", len(customer_churn_df))


print(customer_churn_df.head())


# Display the first few rows of the dataset
#print(customer_churn_df.head())

# Split datasets
X = customer_churn_df.drop("Churn", axis=1)
y = customer_churn_df["Churn"]


# Scale values
scaler_customer_churn = StandardScaler()
X = scaler_customer_churn.fit_transform(X)

#print(X)

X_customer_churn, X_customer_churn, y_customer_churn, y_customer_churn = train_test_split(
    X, y, test_size=0.2, random_state=42
)