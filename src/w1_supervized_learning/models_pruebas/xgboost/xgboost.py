import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score

class GradientBoosted:

    def __init__(self):
        pass

    def cross_validate(self, model, X_train, y_train, cv=5):
        scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1)
        print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores

    def train(self, X_train, X_test, y_train, y_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(X_train_scaled.shape, y_train.shape)

        model = xgb.XGBClassifier(
            eval_metric='mlogloss',
            objective='binary:logistic',
            random_state=42
        )

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0]
        }

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )
        grid_search.fit(X_train_scaled, y_train)

        print(f"Best parameters: {grid_search.best_params_}")

        model = grid_search.best_estimator_

        self.cross_validate(model, X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy}")
