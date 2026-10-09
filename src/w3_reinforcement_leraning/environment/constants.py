"""Constantes que definen el estado y las acciones del MDP Telco."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetentionAction:
    name: str
    acceptance_probability: float
    cost: float = 0.0


# Actions defined with name, acceptance probability and cost
ACTION_DEFINITIONS = (
    RetentionAction("Offer One Year contract", 0.85),
    RetentionAction("Offer Two Year contract", 0.85),
    RetentionAction("Offer 5% discount", 0.95, 0.05),
    RetentionAction("Offer 10% discount", 0.95, 0.10),
    RetentionAction("Offer 15% discount", 0.95, 0.15),
    RetentionAction("Offer Phone Service", 0.90),
    RetentionAction("Offer Multiple Lines", 0.90),
    RetentionAction("Offer DSL Internet", 0.90),
    RetentionAction("Offer Fiber Optic Internet", 0.90),
    RetentionAction("Offer Online Security", 0.90),
    RetentionAction("Offer Online Backup", 0.90),
    RetentionAction("Offer Device Protection", 0.90),
    RetentionAction("Offer Tech Support", 0.90),
    RetentionAction("Offer Streaming TV", 0.90),
    RetentionAction("Offer Streaming Movies", 0.90),
)

# 19 features of the customer, without churn and customer id
RAW_FEATURES = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
)


CATEGORICAL_FEATURES = (
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)

CONTINUOUS_FEATURES = (
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
)

ENGINEERED_FEATURES = ("NewCustomer", "AvgMonthlyCharges")

# Total of features observables. 
OBSERVATION_FEATURES = (
    *RAW_FEATURES, #19
    *ENGINEERED_FEATURES, #2
    "churn_probability", #1
)

# Price approximated for services
SERVICE_PRICES = {
    5: 20.0,
    6: 5.0,
    7: 25.0,
    8: 50.0,
    9: 5.0,
    10: 5.0,
    11: 5.0,
    12: 5.0,
    13: 10.0,
    14: 10.0,
}

INTERNET_DEPENDENT_SERVICES = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)