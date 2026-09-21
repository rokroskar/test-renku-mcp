"""Fit a classifier and write its metrics out — usable as a Renku job.

    python train.py --model "Logistic regression" --n-train 12000 --out models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import classification_report

from mnist_data import find_data_dir, load_split
from mnist_model import DIGITS, MODELS, fit_and_evaluate, subsample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Logistic regression", choices=list(MODELS))
    parser.add_argument("--n-train", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    data_dir = find_data_dir()
    print(f"Reading the Zenodo record from {data_dir}")
    x_train, y_train = load_split(data_dir, "train")
    x_test, y_test = load_split(data_dir, "test")

    x_fit, y_fit = subsample(x_train, y_train, args.n_train, args.seed)
    print(f"Fitting {args.model} on {len(y_fit):,} images")
    result = fit_and_evaluate(args.model, x_fit, y_fit, x_test, y_test, args.seed)

    print(f"Test accuracy: {result.accuracy:.4f}")
    print(
        classification_report(
            result.y_true, result.y_pred, labels=DIGITS, zero_division=0
        )
    )

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.estimator, args.out / "model.joblib")
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "model": result.model_name,
                "n_train": result.n_train,
                "n_test": int(len(result.y_true)),
                "accuracy": result.accuracy,
                "confusion_matrix": result.confusion.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.out / 'model.joblib'} and {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
