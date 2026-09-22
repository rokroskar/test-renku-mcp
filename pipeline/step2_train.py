"""Step 2: fit every classifier/training-size combination.

Consumes step 1's dataset bundle and publishes the fitted estimators. It never
touches the data connector -- if step 1 did not run, this step has nothing to
read, which is exactly the dependency the orchestration has to respect.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from _common import publish, resolve_input

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mnist_model import MODELS, subsample  # noqa: E402

SOURCE = "https://github.com/rokroskar/test-renku-mcp"
TRAIN_SIZES = (1000, 4000, 12000)
SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="local_in", type=Path, default=None)
    parser.add_argument("--pull", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--push", default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    source_dir = resolve_input(args.pull, args.out / "_input", args.local_in)

    data = np.load(source_dir / "dataset.npz")
    x_train = data["x_train"].astype(np.float32) / 255.0
    y_train = data["y_train"].astype(int)
    print(f"Loaded {len(y_train):,} training images")

    estimators = {}
    entries = []
    for model_name in MODELS:
        for n_train in TRAIN_SIZES:
            key = f"{len(entries):02d}"
            x_fit, y_fit = subsample(x_train, y_train, n_train, SEED)

            started = time.perf_counter()
            estimator = MODELS[model_name](SEED)
            estimator.fit(x_fit, y_fit)
            elapsed = time.perf_counter() - started

            estimators[key] = estimator
            entries.append(
                {
                    "key": key,
                    "model": model_name,
                    "n_train": int(len(y_fit)),
                    "fit_seconds": round(elapsed, 1),
                }
            )
            print(f"  {model_name} @ {len(y_fit):>6,}: fitted in {elapsed:.1f}s")

    joblib.dump(estimators, args.out / "estimators.joblib")
    (args.out / "estimators.json").write_text(
        json.dumps({"seed": SEED, "configs": entries}, indent=2) + "\n"
    )

    size = (args.out / "estimators.joblib").stat().st_size / 1e6
    print(f"Wrote {args.out}/estimators.joblib ({size:.1f} MB)")
    publish(args.push, args.out, ["estimators.joblib", "estimators.json"], SOURCE)


if __name__ == "__main__":
    main()
