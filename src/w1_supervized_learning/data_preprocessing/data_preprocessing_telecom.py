# This preprocessing is for the Telecom dataset.

import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv

load_dotenv()

path_dataset_telecom = os.getenv("PATH_DATASET_TELECOM")
telecom_churn_df = pd.read_csv(path_dataset_telecom)

# Remove features (customer_id, pincode and date_of_registration are left out,
# the date is used later in the feature engineering stage)
columns = ["telecom_partner", "gender", "age", "state", "city", "num_dependents",
           "estimated_salary", "calls_made", "sms_sent", "data_used", "churn"]
telecom_churn_df = telecom_churn_df[columns]

# Encode telecom_partner
telecom_churn_df["telecom_partner"] = telecom_churn_df["telecom_partner"].map({"Airtel": 0, "BSNL": 1, "Reliance Jio": 2, "Vodafone": 3})

# Encode gender
telecom_churn_df["gender"] = telecom_churn_df["gender"].map({"F": 0, "M": 1})

# Encode state
telecom_churn_df["state"] = telecom_churn_df["state"].map({"Andhra Pradesh": 0, "Arunachal Pradesh": 1, "Assam": 2,
    "Bihar": 3, "Chhattisgarh": 4, "Goa": 5, "Gujarat": 6, "Haryana": 7, "Himachal Pradesh": 8, "Jharkhand": 9,
    "Karnataka": 10, "Kerala": 11, "Madhya Pradesh": 12, "Maharashtra": 13, "Manipur": 14, "Meghalaya": 15,
    "Mizoram": 16, "Nagaland": 17, "Odisha": 18, "Punjab": 19, "Rajasthan": 20, "Sikkim": 21, "Tamil Nadu": 22,
    "Telangana": 23, "Tripura": 24, "Uttar Pradesh": 25, "Uttarakhand": 26, "West Bengal": 27})

# Encode city
telecom_churn_df["city"] = telecom_churn_df["city"].map({"Bangalore": 0, "Chennai": 1, "Delhi": 2, "Hyderabad": 3, "Kolkata": 4, "Mumbai": 5})

print("Datos sin borrar NaN:", len(telecom_churn_df))

# Drop NaN values
telecom_churn_df = telecom_churn_df.dropna()

print("Datos borrados los NaN:", len(telecom_churn_df))

print(telecom_churn_df.head())

# Display the first few rows of the dataset
#print(telecom_churn_df.head())

# Split datasets
X = telecom_churn_df.drop("churn", axis=1)
y = telecom_churn_df["churn"]

# Scale values
scaler_telecom_churn = StandardScaler()
X = scaler_telecom_churn.fit_transform(X)

#print(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)