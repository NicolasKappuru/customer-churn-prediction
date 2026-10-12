"""Constantes que definen el estado y las acciones del MDP Telco."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetentionAction:
    name: str
    acceptance_probability: float
    cost: float = 0.0


# Actions defined with name, acceptance probability and cost
ACTION_DEFINITIONS = (
    RetentionAction("Offer One Year contract", 0.60),
    RetentionAction("Offer Secure plan", 0.80),
    RetentionAction("Offer billing and payment change", 0.95),
    RetentionAction("Offer 30% discount", 0.95, 0.10),
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

SECURE_PLAN_SERVICES = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
)

SECURE_PLAN_SERVICE_PRICE = 5.0
DSL_SERVICE_PRICE = 25.0

SECURE_PLAN_BENEFIT_PER_SERVICE = 0.05
OTHER_ACTION_BENEFIT = 0.0

DISCOUNT_PERCENTAGE = 0.30