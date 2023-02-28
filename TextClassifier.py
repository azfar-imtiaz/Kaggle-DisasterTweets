import numpy as np
from sklearn.svm import SVC


class TextClassifier:
    def __init__(self) -> None:
        self.model = SVC()

    def train_model(self, X_train: np.array, y_train: np.array) -> SVC:
        self.model.fit(X_train, y_train)

    def get_predictions(self, X_test: np.array) -> np.array:
        return self.model.predict(X_test)
