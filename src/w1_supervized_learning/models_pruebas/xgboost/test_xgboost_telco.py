from src.w1_supervized_learning.data_preprocessing.data_preprocessing_telco import PreprocessingTelco
from src.w1_supervized_learning.models_pruebas.xgboost.xgboost import GradientBoosted

telco = PreprocessingTelco()
X_train, X_test, y_train, y_test = telco.preprocess()

gradient_boosted = GradientBoosted()
gradient_boosted.train(X_train, X_test, y_train, y_test)
