from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

class RandomForest:

    def __init__(self):
        pass

    def cross_validate(self, model, X_train, y_train, cv=5):
        scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1)
        print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores

    def train(self, X_train, X_test, y_train, y_test):

        clf = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(
            clf,
            param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")

        clf = grid_search.best_estimator_

        self.cross_validate(clf, X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")