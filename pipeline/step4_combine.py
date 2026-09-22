"""Fan-in step: score every branch's estimators into one results bundle.

Takes the dataset from step 1 and one artifact per training branch. It refuses
to run unless every branch it was told to expect is present, so a branch that
failed or has not finished yet cannot be silently dropped from the results.
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
    parser.add_argument(
        "--branch-in", type=Path, action="append", default=[],
        help="A branch output directory; repeatable.",
    )
    parser.add_argument(
        "--branch-pull", action="append", default=[],
        help="A branch OCI reference; repeatable.",
    )
    parser.add_argument(
        "--expect-branches", type=int, default=None,
        help="Fail unless exactly this many branches were supplied.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--push", default=None)
    args = parser.parse_args()

    sources = [("in", p) for p in args.branch_in]
    sources += [("pull", r) for r in args.branch_pull]
    if not sources:
        raise SystemExit("at least one --branch-in or --branch-pull is required")
    if args.expect_branches is not None and len(sources) != args.expect_branches:
        raise SystemExit(
            f"expected {args.expect_branches} branches, got {len(sources)} -- "
            "refusing to publish a partial result"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_input(args.data_pull, args.out / "_data", args.data_in)
    data = np.load(data_dir / "dataset.npz")
    x_test = data["x_test"].astype(np.float32) / 255.0
    y_test = data["y_test"].astype(int)

    collected = []
    for index, (kind, value) in enumerate(sources):
        into = args.out / f"_branch{index}"
        branch_dir = resolve_input(
            value if kind == "pull" else None,
            into,
            Path(value) if kind == "in" else None,
        )
        spec = json.loads((branch_dir / "branch.json").read_text())
        estimators = joblib.load(branch_dir / "estimators.joblib")
        print(f"  branch '{spec['family']}': {len(spec['configs'])} configs")
        for entry in spec["configs"]:
            collected.append((entry, estimators[entry["key"]]))

    # Deterministic order regardless of which branch finished first.
    collected.sort(key=lambda pair: (pair[0]["model"], pair[0]["n_train"]))

    arrays = {"images": data["x_test"], "labels": data["y_test"]}
    configs = []
    for index, (entry, estimator) in enumerate(collected):
        key = f"{index:02d}"
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
                "key": key,
                "model": entry["model"],
                "n_train": entry["n_train"],
                "fit_seconds": entry["fit_seconds"],
                "score_kind": score_kind,
                "accuracy": accuracy,
                "confusion": confusion_matrix(
                    y_test, predictions, labels=DIGITS
                ).tolist(),
            }
        )
        print(f"  {entry['model']} @ {entry['n_train']:>6,}: accuracy {accuracy:.4f}")

    np.savez_compressed(args.out / "results.npz", **arrays)
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "doi": "10.5281/zenodo.4697906",
                "seed": 0,
                "digits": DIGITS,
                "n_test": int(len(y_test)),
                "branches": len(sources),
                "configs": configs,
            },
            indent=2,
        )
        + "\n"
    )

    size = (args.out / "results.npz").stat().st_size / 1e6
    print(f"Combined {len(sources)} branches into {len(configs)} configs")
    print(f"Wrote {args.out}/results.npz ({size:.1f} MB)")
    publish(args.push, args.out, ["results.npz", "manifest.json"], SOURCE)


if __name__ == "__main__":
    main()
