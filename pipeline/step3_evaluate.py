"""Step 3: score the fitted estimators and build the app's results bundle.

Consumes both earlier steps -- the dataset from step 1 and the estimators from
step 2 -- and emits exactly the layout mnist_artifacts.Artifacts reads, so the
app can be pointed at this bundle by changing RESULTS_REFERENCE alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

from _common import publish, resolve_input

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mnist_model import DIGITS  # noqa: E402

SOURCE = "https://github.com/rokroskar/test-renku-mcp"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-in", type=Path, default=None)
    parser.add_argument("--data-pull", default=None)
    parser.add_argument("--models-in", type=Path, default=None)
    parser.add_argument("--models-pull", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--push", default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_input(args.data_pull, args.out / "_data", args.data_in)
    models_dir = resolve_input(args.models_pull, args.out / "_models", args.models_in)

    data = np.load(data_dir / "dataset.npz")
    x_test = data["x_test"].astype(np.float32) / 255.0
    y_test = data["y_test"].astype(int)

    estimators = joblib.load(models_dir / "estimators.joblib")
    spec = json.loads((models_dir / "estimators.json").read_text())

    arrays = {
        "images": data["x_test"],
        "labels": data["y_test"],
    }
    configs = []
    for entry in spec["configs"]:
        key = entry["key"]
        estimator = estimators[key]
        predictions = estimator.predict(x_test)

        if hasattr(estimator, "predict_proba"):
            scores, score_kind = estimator.predict_proba(x_test), "probability"
        else:
            scores, score_kind = estimator.decision_function(x_test), "margin"

        accuracy = float((predictions == y_test).mean())
        arrays[f"pred_{key}"] = predictions.astype(np.uint8)
        arrays[f"score_{key}"] = scores.astype(np.float32)

        configs.append(
            {
                **entry,
                "score_kind": score_kind,
                "accuracy": accuracy,
                "confusion": confusion_matrix(
                    y_test, predictions, labels=DIGITS
                ).tolist(),
            }
        )
        print(
            f"  {entry['model']} @ {entry['n_train']:>6,}: "
            f"accuracy {accuracy:.4f}"
        )

    np.savez_compressed(args.out / "results.npz", **arrays)
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "doi": "10.5281/zenodo.4697906",
                "seed": spec["seed"],
                "digits": DIGITS,
                "n_test": int(len(y_test)),
                "configs": configs,
            },
            indent=2,
        )
        + "\n"
    )

    size = (args.out / "results.npz").stat().st_size / 1e6
    print(f"Wrote {args.out}/results.npz ({size:.1f} MB)")
    publish(args.push, args.out, ["results.npz", "manifest.json"], SOURCE)


if __name__ == "__main__":
    main()
