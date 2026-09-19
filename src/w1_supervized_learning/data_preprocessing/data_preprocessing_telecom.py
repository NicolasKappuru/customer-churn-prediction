# This preprocessing is for the Telecom dataset.

import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv

load_dotenv()

class PreprocessingTelecom:

    def __init__(self):
        pass

    def preprocess(self):

        path_dataset_telecom = os.getenv("PATH_DATASET_TELECOM")
        telecom_churn_df = pd.read_csv(path_dataset_telecom)

        # Remove features (customer_id, pincode and date_of_registration are left out,
        # the date is used later in the feature engineering stage)
        columns = ["telecom_partner", "gender", "age", "state", "city", "num_dependents",
                "estimated_salary", "calls_made", "sms_sent", "data_used", "churn"]
        telecom_churn_df = telecom_churn_df[columns]

        # Encode binary features
        telecom_churn_df["gender"] = telecom_churn_df["gender"].map({"F": 0, "M": 1})

        # Encode with one hot encoding the features with more than two categories
        one_hot_columns_telecom = ["telecom_partner", "state", "city"]
        telecom_churn_df = pd.get_dummies(telecom_churn_df, columns=one_hot_columns_telecom, dtype=int)

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

        return X_train, X_test, y_train, y_test
    
telecom = PreprocessingTelecom()
telecom.preprocess()