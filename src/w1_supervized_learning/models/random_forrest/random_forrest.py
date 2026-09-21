print("Here come the random forrest")

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from w1_supervized_learning.data_preprocessing.data_preprocessing_telco import PreprocessingTelco
from w1_supervized_learning.data_preprocessing.data_preprocessing_telecom import PreprocessingTelecom

print("Here come xgboost")

class RandomForestModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train, params):
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        balanced_acc = balanced_accuracy_score(y_test, preds)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'balanced_accuracy': balanced_acc
        }
    
    def train_k_fold(self, X_train, y_train, params, n_splits=5):
        #Here I create a StratifiedKFold object with 5 splits, shuffling the data and setting the random state to 42 for reproducibility
        #The idea is that every fold has the same proprtion of classes(churn and not churn) as the original dataset, 
        # this is important because the dataset is imbalanced

        #We do ir in this way to avoid overfitting and to get a better estimate of the model's performance on unseen data
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracy_scores = []
        f1_scores = []
        balanced_accuracy_scores = []

        for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), start=1):
            print(f"Fold {fold}:")
            X_fold_train = X_train[train_index]
            X_fold_val = X_train[val_index]
            y_fold_train = y_train.iloc[train_index]
            y_fold_val = y_train.iloc[val_index]

            self.train(X_fold_train, y_fold_train, params)
            preds = self.predict(X_fold_val)

            accuracy = accuracy_score(y_fold_val, preds)
            f1 = f1_score(y_fold_val, preds)
            balanced_acc = balanced_accuracy_score(y_fold_val, preds)

            accuracy_scores.append(accuracy)
            f1_scores.append(f1)
            balanced_accuracy_scores.append(balanced_acc)

            print(f'  Accuracy: {accuracy:.4f}')
            print(f'  F1: {f1:.4f}')
            print(f'  Balanced Accuracy: {balanced_acc:.4f}')

        print('\nAverage Results:')
        print('Accuracy:', sum(accuracy_scores) / len(accuracy_scores))
        print('F1:', sum(f1_scores) / len(f1_scores))
        print('Balanced Accuracy:', sum(balanced_accuracy_scores) / len(balanced_accuracy_scores))
        
telco = PreprocessingTelco() 
X_train_telco, X_test_telco, y_train_telco, y_test_telco = telco.preprocess() #Here I get the preprocessed data from the PreprocessingTelco class


telecom = PreprocessingTelecom()
X_train_telecom, X_test_telecom, y_train_telecom, y_test_telecom = telecom.preprocess() #Here I get the preprocessed data from the PreprocessingTelecom class

params = {
    'n_estimators': 100, # The number of trees is set to 100, which is a common value for Random Forest
    'max_depth': 5, # The maximum depth of the trees is set to 5, which is a common value for Random Forest this is used to avoid overfitting
    'random_state': 42, # The random state is set to 42, which is a common value for reproducibility
    "class_weight": "balanced"
}


xgb_model_telco = RandomForestModel()
xgb_model_telco.train_k_fold(X_train_telco, y_train_telco, params)


xgb_model_telcom = RandomForestModel()
xgb_model_telcom.train_k_fold(X_train_telecom, y_train_telecom, params)



