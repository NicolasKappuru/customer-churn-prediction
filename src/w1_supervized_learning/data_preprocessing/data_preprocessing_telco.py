# This preprocessing is for the Telco dataset. 
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

load_dotenv()

class PreprocessingTelco:

    def __init__(self):
        path_dataset_telco = os.getenv("PATH_DATASET_TELCO")
        self.telco_churn_df = pd.read_csv(path_dataset_telco)


    def preprocess(self):
        # Preprocessing data from dataset

        self.select_features()
        self.encode()
        self.drop_nan()
        self.make_feature_engineering()

        return self.split_dataset()

    
    def select_features(self):
        # Select features

        # Not customerID (unique per row, useless for the model)

        columns = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "PaperlessBilling", "PaymentMethod",
                "MonthlyCharges", "TotalCharges", "Churn"]
        self.telco_churn_df = self.telco_churn_df[columns]


    def encode(self):
        # Encode target
        self.telco_churn_df["Churn"] = self.telco_churn_df["Churn"].map({"Yes": 1, "No": 0})

        categorical_columns_telco = ["gender", "Partner", "Dependents", "PhoneService",
                    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaperlessBilling", "PaymentMethod"]
        self.telco_churn_df = pd.get_dummies(
            self.telco_churn_df,
            columns=categorical_columns_telco,
            dtype=int,
        )


    def drop_nan(self):
        # Drop NaN values

        # TotalCharges has some attributes in blank
        # Blank TotalCharges belong to new customers (tenure 0), so we fill them with 0
        self.telco_churn_df["TotalCharges"] = pd.to_numeric(
            self.telco_churn_df["TotalCharges"], errors="coerce"
        ).fillna(0)

        self.telco_churn_df = self.telco_churn_df.dropna()


    def make_feature_engineering(self):
        # Light feature engineering

        # NewCustomer: customers with tenure 0 are more likely to churn
        self.telco_churn_df["NewCustomer"] = (self.telco_churn_df["tenure"] == 0).astype(int)

        # AvgMonthlyCharges: keeps the average billing per month
        self.telco_churn_df["AvgMonthlyCharges"] = np.where(
            self.telco_churn_df["tenure"] > 0,
            self.telco_churn_df["TotalCharges"] / self.telco_churn_df["tenure"],
            0
        )


    def split_dataset(self):
        # Split datasets
        X = self.telco_churn_df.drop("Churn", axis=1)
        y = self.telco_churn_df["Churn"]
        self.feature_names = X.columns.tolist()

        # XGBoost is a tree-based model, so scaling is not needed.
        # We use stratify because the dataset is imbalanced.
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        

        return X_train, X_test, y_train, y_test
