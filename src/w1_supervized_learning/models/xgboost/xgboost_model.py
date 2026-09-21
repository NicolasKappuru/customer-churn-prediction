from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from w1_supervized_learning.data_preprocessing.data_preprocessing_telco import PreprocessingTelco

print("Here come xgboost")

telco = PreprocessingTelco() 
X_train, X_test, y_train, y_test = telco.preprocess() #Here I get the preprocessed data from the PreprocessingTelco class

params = {
    'objective': 'binary:logistic', # The problem is determinate churn or not churn, so it's a binary classification problem
    'eval_metric': 'logloss', # The evaluation metric is logloss, which is suitable for binary classification problems
    'learning_rate': 0.1, # The learning rate is set to 0.1, which is a common value for XGBoost
    'max_depth': 5, # The maximum depth of the trees is set to 5, which is a common value for XGBoost this is used to avoid overfitting
    'seed': 42, # The seed is set to 42, which is a common value for reproducibility
    'n_estimators': 100, # The number of trees is set to 100, which is a common value for XGBoost
}

#Here I create a StratifiedKFold object with 5 splits, shuffling the data and setting the random state to 42 for reproducibility
#The idea is that every fold has the same proprtion of classes(churn and not churn) as the original dataset, 
# this is important because the dataset is imbalanced

#We do ir in this way to avoid overfitting and to get a better estimate of the model's performance on unseen data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

accuracy_scores = []
f1_scores = []
balanced_accuracy_scores = []

for fold, (train_index, val_index) in enumerate(
    skf.split(X_train, y_train), start=1
):

    print(f"Fold {fold}:")

    X_fold_train = X_train[train_index]
    X_fold_val = X_train[val_index]

    y_fold_train = y_train.iloc[train_index]
    y_fold_val = y_train.iloc[val_index]

    xgb_model = XGBClassifier(**params)

    xgb_model.fit(
        X_fold_train,
        y_fold_train
    )

    preds = xgb_model.predict(X_fold_val)

    accuracy = accuracy_score(y_fold_val, preds)
    f1 = f1_score(y_fold_val, preds)
    balanced_acc = balanced_accuracy_score(y_fold_val, preds)

    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    balanced_accuracy_scores.append(balanced_acc)

    print(f'  Accuracy: {accuracy:.4f}')
    print(f'  F1: {f1:.4f}')
    print(f'  Balanced Accuracy: {balanced_acc:.4f}')

print('\nResultados promedio:')
print('Accuracy:', sum(accuracy_scores) / len(accuracy_scores))
print('F1:', sum(f1_scores) / len(f1_scores))
print(
    'Balanced Accuracy:',
    sum(balanced_accuracy_scores) / len(balanced_accuracy_scores)
)