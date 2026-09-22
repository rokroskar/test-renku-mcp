"""Read the frozen results the CI build produced with precompute.py."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

SEARCH = ("artifacts", "/app/artifacts")


@dataclass(frozen=True)
class Config:
    """One frozen classifier/training-size combination."""

    key: str
    model: str
    n_train: int
    score_kind: str
    accuracy: float
    confusion: np.ndarray
    fit_seconds: float

    @property
    def scores_are_probabilities(self) -> bool:
        return self.score_kind == "probability"


class Artifacts:
    def __init__(self, directory: Path):
        self.directory = directory
        manifest = json.loads((directory / "manifest.json").read_text())
        self.doi: str = manifest["doi"]
        self.seed: int = manifest["seed"]
        self.digits: list[int] = manifest["digits"]
        self.n_test: int = manifest["n_test"]
        self._npz = np.load(directory / "results.npz")
        self.configs = [
            Config(
                key=entry["key"],
                model=entry["model"],
                n_train=entry["n_train"],
                score_kind=entry["score_kind"],
                accuracy=entry["accuracy"],
                confusion=np.array(entry["confusion"]),
                fit_seconds=entry["fit_seconds"],
            )
            for entry in manifest["configs"]
        ]

    @cached_property
    def images(self) -> np.ndarray:
        """Test images as uint8, (n_test, 784)."""
        return self._npz["images"]

    @cached_property
    def labels(self) -> np.ndarray:
        return self._npz["labels"].astype(int)

    def predictions(self, config: Config) -> np.ndarray:
        return self._npz[f"pred_{config.key}"].astype(int)

    def scores(self, config: Config) -> np.ndarray:
        return self._npz[f"score_{config.key}"]

    @property
    def models(self) -> list[str]:
        seen = {}
        for config in self.configs:
            seen.setdefault(config.model, None)
        return list(seen)

    def sizes_for(self, model: str) -> list[int]:
        return sorted(c.n_train for c in self.configs if c.model == model)

    def find(self, model: str, n_train: int) -> Config:
        for config in self.configs:
            if config.model == model and config.n_train == n_train:
                return config
        raise KeyError(f"no frozen result for {model} @ {n_train}")


def find_artifacts_dir() -> Path:
    override = os.environ.get("ARTIFACTS_DIR")
    candidates = [Path(override)] if override else [Path(p) for p in SEARCH]
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError(
        "No precomputed results found. Run `python precompute.py` or set "
        "ARTIFACTS_DIR."
    )


def per_digit_metrics(confusion: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall, F1 and support straight from the confusion matrix.

    Doing the arithmetic here keeps scikit-learn (and scipy) out of the app
    image — the app never fits anything.
    """
    true_positive = np.diag(confusion).astype(float)
    predicted = confusion.sum(axis=0).astype(float)
    actual = confusion.sum(axis=1).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, true_positive / predicted, 0.0)
        recall = np.where(actual > 0, true_positive / actual, 0.0)
        denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * precision * recall / denominator, 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": actual.astype(int),
    }
