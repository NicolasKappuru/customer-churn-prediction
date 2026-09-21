from src.w1_supervized_learning.data_preprocessing.data_preprocessing_telecom import PreprocessingTelecom
from src.w1_supervized_learning.models_pruebas.xgboost.xgboost import GradientBoosted

telecom = PreprocessingTelecom()
X_train, X_test, y_train, y_test = telecom.preprocess()

gradient_boosted = GradientBoosted()
gradient_boosted.train(X_train, X_test, y_train, y_test)