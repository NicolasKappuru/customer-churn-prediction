# This preprocessing is for the Telco dataset. 

import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv

load_dotenv()

class PreprocessingTelco:

    def __init__(self):
        pass

    def preprocess(self):

        path_dataset_telco = os.getenv("PATH_DATASET_TELCO")
        telco_churn_df = pd.read_csv(path_dataset_telco)

        # Not PaperBilling, PaymentMethod,

        # Remove features
        columns = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "MonthlyCharges", "TotalCharges", "Churn"]
        telco_churn_df = telco_churn_df[columns]

        # Encode binary features

        # Encode gender
        telco_churn_df["gender"] = telco_churn_df["gender"].map({"Male": 0, "Female": 1})

        # Encode Partner
        telco_churn_df["Partner"] = telco_churn_df["Partner"].map({"No": 0, "Yes": 1})

        # Encode Dependents
        telco_churn_df["Dependents"] = telco_churn_df["Dependents"].map({"No": 0, "Yes": 1})

        # Encode PhoneService
        telco_churn_df["PhoneService"] = telco_churn_df["PhoneService"].map({"No": 0, "Yes": 1})

        # Encode Churn
        telco_churn_df["Churn"] = telco_churn_df["Churn"].map({"No": 0, "Yes": 1})

        # Encode with one hot encoding the features with more than two categories
        one_hot_columns_telco = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                                "Contract"]
        telco_churn_df = pd.get_dummies(telco_churn_df, columns=one_hot_columns_telco, dtype=int)

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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test
    
telco = PreprocessingTelco()
telco.preprocess()