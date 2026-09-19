# This preprocessing is for the Telecom dataset.
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv

load_dotenv()

class PreprocessingTelecom:

    def __init__(self):
        path_dataset_telecom = os.getenv("PATH_DATASET_TELECOM")
        self.telecom_churn_df = pd.read_csv(path_dataset_telecom)


    def preprocess(self):
        # Preprocessing data from dataset
                
        self.select_features()
        self.make_feature_engineering()
        self.encode()
        self.drop_nan()
        
        print(self.telecom_churn_df.head())

        return self.split_dataset()
    
    def select_features(self):
        # Select features

        # customer_id, pincode and date_of_registration are left out,
        # the date is used later in the feature engineering stage

        columns = ["telecom_partner", "gender", "age", "state", "city", "num_dependents",
                "estimated_salary", "calls_made", "sms_sent", "data_used", "churn"]
        self.telecom_churn_df = self.telecom_churn_df[columns]


    def make_feature_engineering(self):
        # Feature Engineering

        pass


    def encode(self):
        # Encode binary features

        # Encode gender
        self.telecom_churn_df["gender"] = self.telecom_churn_df["gender"].map({"F": 0, "M": 1})

        # Encode with one hot encoding the features with more than two categories
        one_hot_columns_telecom = ["telecom_partner", "state", "city"]
        self.telecom_churn_df = pd.get_dummies(self.telecom_churn_df, columns=one_hot_columns_telecom, dtype=int)
        

    def drop_nan(self):
        # Drop NaN values

        print("Datos sin borrar NaN:", len(self.telecom_churn_df))

        self.telecom_churn_df = self.telecom_churn_df.dropna()

        print("Datos borrados los  NaN:", len(self.telecom_churn_df))


    def split_dataset(self):
        # Split datasets
        X = self.telecom_churn_df.drop("churn", axis=1)
        y = self.telecom_churn_df["churn"]


        # Scale values
        scaler_telecom_churn = StandardScaler()
        X = scaler_telecom_churn.fit_transform(X)

        #print(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test


telecom = PreprocessingTelecom()
telecom.preprocess()