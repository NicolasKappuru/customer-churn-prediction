from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

class RandomForest:

    def __init__(self):
        pass

    def train(self, X_train, X_test, y_train, y_test):

        clf = RandomForestClassifier()

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
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")

        clf = grid_search.best_estimator_

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")