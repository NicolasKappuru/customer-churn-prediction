from src.w1_supervized_learning.data_preprocessing.data_preprocessing_telecom import PreprocessingTelecom
from src.w1_supervized_learning.models_pruebas.random_forest.random_forest import RandomForest

telecom = PreprocessingTelecom()
X_train, X_test, y_train, y_test = telecom.preprocess()

random_forest = RandomForest()
random_forest.train(X_train, X_test, y_train, y_test)
