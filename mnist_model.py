"""Digit classifiers and their evaluation on the Zenodo MNIST subset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

MODELS = {
    "Logistic regression": lambda seed: LogisticRegression(
        max_iter=200, random_state=seed
    ),
    "Linear SVM": lambda seed: LinearSVC(C=0.01, random_state=seed),
    "Neural net (1x64)": lambda seed: MLPClassifier(
        hidden_layer_sizes=(64,), max_iter=60, random_state=seed
    ),
}

DIGITS = list(range(10))


@dataclass
class Result:
    """A fitted model together with its test-set predictions."""

    model_name: str
    n_train: int
    estimator: object
    y_true: np.ndarray
    y_pred: np.ndarray
    scores: np.ndarray | None  # (n_samples, 10) probabilities, when available

    @property
    def accuracy(self) -> float:
        return float((self.y_true == self.y_pred).mean())

    @property
    def confusion(self) -> np.ndarray:
        return confusion_matrix(self.y_true, self.y_pred, labels=DIGITS)

    @property
    def errors(self) -> np.ndarray:
        """Indices of the misclassified test images."""
        return np.flatnonzero(self.y_true != self.y_pred)


def subsample(
    features: np.ndarray, labels: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Take a random, class-proportional-in-expectation subset."""
    if n >= len(labels):
        return features, labels
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(labels), size=n, replace=False)
    return features[picked], labels[picked]


def fit_and_evaluate(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 0,
) -> Result:
    estimator = MODELS[model_name](seed)
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)

    scores = None
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(x_test)

    return Result(
        model_name=model_name,
        n_train=len(y_train),
        estimator=estimator,
        y_true=y_test,
        y_pred=y_pred,
        scores=scores,
    )
