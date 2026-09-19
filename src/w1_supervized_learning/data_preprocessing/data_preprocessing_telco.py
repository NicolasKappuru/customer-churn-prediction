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
customer_churn_df = pd.read_csv(path_dataset_telco)

# Not PaperBilling, PaymentMethod,

# Remove features
columns = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
           "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
           "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
           "Contract", "MonthlyCharges", "TotalCharges", "Churn"]
customer_churn_df = customer_churn_df[columns]

# Encode binary features

# Encode gender
customer_churn_df["gender"] = customer_churn_df["gender"].map({"Male": 0, "Female": 1})

# Encode Partner
customer_churn_df["Partner"] = customer_churn_df["Partner"].map({"No": 0, "Yes": 1})

# Encode Dependents
customer_churn_df["Dependents"] = customer_churn_df["Dependents"].map({"No": 0, "Yes": 1})

# Encode PhoneService
customer_churn_df["PhoneService"] = customer_churn_df["PhoneService"].map({"No": 0, "Yes": 1})

# Encode Churn
customer_churn_df["Churn"] = customer_churn_df["Churn"].map({"No": 0, "Yes": 1})

# Encode with one hot encoding the features with more than two categories
one_hot_columns_telco = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                         "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                         "Contract"]
customer_churn_df = pd.get_dummies(customer_churn_df, columns=one_hot_columns_telco, dtype=int)

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)