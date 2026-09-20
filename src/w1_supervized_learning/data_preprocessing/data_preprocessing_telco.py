# This preprocessing is for the Telco dataset. 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv

load_dotenv()

class PreprocessingTelco:

    def __init__(self):
        path_dataset_telco = os.getenv("PATH_DATASET_TELCO")
        self.telco_churn_df = pd.read_csv(path_dataset_telco)


    def preprocess(self):
        # Preprocessing data from dataset

        self.select_features()
        self.make_feature_engineering()
        self.clean_data()
        print(self.telco_churn_df[['MultipleLines','InternetService', 'StreamingTV', 'NumberServices']].head())
        self.encode()
        self.drop_nan()

        return self.split_dataset()

    
    def select_features(self):
        # Select features

        # Not PaperBilling, PaymentMethod,

        columns = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "MonthlyCharges", "TotalCharges", "Churn"]
        self.telco_churn_df = self.telco_churn_df[columns]


    def make_feature_engineering(self):
        # Feature Engineering

        service_columns = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        self.telco_churn_df["NumberServices"] = (
            self.telco_churn_df[service_columns]
            .eq("Yes")
            .sum(axis=1)
        )

        self.telco_churn_df["NumberServices"] += (
        self.telco_churn_df["InternetService"]
            .ne("No")
            .astype(int)
        )

    def clean_data(self):
        # Clean data

        # Deleting redundant data like No phone service and No internet service

        internet_service_columns = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        self.telco_churn_df[internet_service_columns] = (
            self.telco_churn_df[internet_service_columns]
            .replace("No internet service", "No")
        )

        self.telco_churn_df["MultipleLines"] = (
            self.telco_churn_df["MultipleLines"]
            .replace("No phone service", "No")
        )




    def encode(self):
        # Encode binary features

        # Encode gender
        self.telco_churn_df["gender"] = self.telco_churn_df["gender"].map({"Male": 0, "Female": 1})

        # Encode binary service features with Yes and No options
        binary_columns_telco = ["Partner", "Dependents", "PhoneService", "MultipleLines",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                    "StreamingTV", "StreamingMovies", "Churn"]
        self.telco_churn_df[binary_columns_telco] = self.telco_churn_df[binary_columns_telco].replace({"No": 0, "Yes": 1})

        # Encode with one hot encoding the features with more than two categories
        one_hot_columns_telco = ["InternetService", "Contract"]
        self.telco_churn_df = pd.get_dummies(self.telco_churn_df, columns=one_hot_columns_telco, dtype=int)
        

    def drop_nan(self):
        # Drop NaN values

        # TotalCharges has some attributes in blank
        # Clean TotalCharges
        self.telco_churn_df["TotalCharges"] = pd.to_numeric(
            self.telco_churn_df["TotalCharges"], errors="coerce"
        )

        print("Datos sin borrar NaN:", len(self.telco_churn_df))

        self.telco_churn_df = self.telco_churn_df.dropna()

        print("Datos borrados los  NaN:", len(self.telco_churn_df))


    def split_dataset(self):
        # Split datasets
        X = self.telco_churn_df.drop("Churn", axis=1)
        y = self.telco_churn_df["Churn"]


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