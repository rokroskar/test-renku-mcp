"""Fan-out step: fit ONE model family across all training sizes.

Three of these run concurrently, one per family, each on its own launcher --
Renku runs a job per launcher, so concurrency means more launchers. Each branch
publishes its own artifact; step4_combine fans them back in.
"""

from __future__ import annotations

import argparse
import json
import re
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


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


FAMILY_BY_SLUG = {slug(name): name for name in MODELS}


def resolve_family(value: str) -> str:
    """Accept either the display name or its slug.

    Every display name contains a space, and these commands travel through a
    JSON tool argument into `bash -c` inside the container -- a quoting chain
    that has already silently split one. A slug has no space to lose.
    """
    if value in MODELS:
        return value
    family = FAMILY_BY_SLUG.get(slug(value))
    if family is None:
        raise SystemExit(
            f"unknown family {value!r}; choose one of "
            + ", ".join(f"{s} ({n})" for s, n in FAMILY_BY_SLUG.items())
        )
    return family


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        required=True,
        help="Model family, as a display name or a slug such as linear-svm.",
    )
    parser.add_argument("--in", dest="local_in", type=Path, default=None)
    parser.add_argument("--pull", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--push", default=None)
    args = parser.parse_args()
    family = resolve_family(args.family)

    args.out.mkdir(parents=True, exist_ok=True)
    source_dir = resolve_input(args.pull, args.out / "_input", args.local_in)

    data = np.load(source_dir / "dataset.npz")
    x_train = data["x_train"].astype(np.float32) / 255.0
    y_train = data["y_train"].astype(int)
    print(f"Branch '{family}' on {len(y_train):,} training images")

    estimators = {}
    entries = []
    for n_train in TRAIN_SIZES:
        key = f"{slug(family)}-{n_train}"
        x_fit, y_fit = subsample(x_train, y_train, n_train, SEED)

        started = time.perf_counter()
        estimator = MODELS[family](SEED)
        estimator.fit(x_fit, y_fit)
        elapsed = time.perf_counter() - started

        estimators[key] = estimator
        entries.append(
            {
                "key": key,
                "model": family,
                "n_train": int(len(y_fit)),
                "fit_seconds": round(elapsed, 1),
            }
        )
        print(f"  {family} @ {len(y_fit):>6,}: fitted in {elapsed:.1f}s")

    joblib.dump(estimators, args.out / "estimators.joblib")
    (args.out / "branch.json").write_text(
        json.dumps(
            {"family": family, "seed": SEED, "configs": entries}, indent=2
        )
        + "\n"
    )

    size = (args.out / "estimators.joblib").stat().st_size / 1e6
    print(f"Wrote {args.out}/estimators.joblib ({size:.1f} MB)")
    publish(args.push, args.out, ["estimators.joblib", "branch.json"], SOURCE)


if __name__ == "__main__":
    main()
