"""Fit every classifier/training-size combination and freeze the results.

Run in CI, not in the app. The app serves anonymous visitors from several
replicas, and each replica would otherwise re-read ~130 MB of TSV and refit on
one core before it could render anything. Freezing the predictions here turns
a pod's cold start into loading ~10 MB from its own filesystem.

    python precompute.py --out artifacts \
        --push ghcr.io/rokroskar/test-renku-mcp/model:latest

Run as a Renku job, it reads the Zenodo record straight from the mounted data
connector and publishes the results to the GitHub container registry, where the
app pulls them anonymously. X_train is deliberately absent from the output: the
app only ever displays test images.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import oci

from mnist_data import find_data_dir, load_split
from mnist_model import DIGITS, MODELS, fit_and_evaluate, subsample

TRAIN_SIZES = (1000, 4000, 12000)
SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--push",
        default=None,
        help="OCI reference to publish the results to, e.g. "
        "ghcr.io/owner/repo/model:latest",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=Path("/secrets/ghcr_token"),
        help="File holding a GitHub token with write:packages. Falls back to "
        "the GITHUB_TOKEN environment variable.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or find_data_dir()
    print(f"Reading the Zenodo record from {data_dir}")
    x_train, y_train = load_split(data_dir, "train")
    x_test, y_test = load_split(data_dir, "test")

    arrays: dict[str, np.ndarray] = {
        # Pixels are 0-255 integers; uint8 costs a quarter of float32 and the
        # app only needs them to draw thumbnails.
        "images": np.rint(x_test * 255).astype(np.uint8),
        "labels": y_test.astype(np.uint8),
    }
    manifest = []

    for model_name in MODELS:
        for n_train in TRAIN_SIZES:
            key = f"{len(manifest):02d}"
            x_fit, y_fit = subsample(x_train, y_train, n_train, SEED)

            started = time.perf_counter()
            result = fit_and_evaluate(
                model_name, x_fit, y_fit, x_test, y_test, SEED
            )
            elapsed = time.perf_counter() - started

            if result.scores is not None:
                scores, score_kind = result.scores, "probability"
            else:
                scores = result.estimator.decision_function(x_test)
                score_kind = "margin"

            arrays[f"pred_{key}"] = result.y_pred.astype(np.uint8)
            arrays[f"score_{key}"] = scores.astype(np.float32)

            manifest.append(
                {
                    "key": key,
                    "model": model_name,
                    "n_train": int(len(y_fit)),
                    "score_kind": score_kind,
                    "accuracy": result.accuracy,
                    "confusion": result.confusion.tolist(),
                    "fit_seconds": round(elapsed, 1),
                }
            )
            print(
                f"  {model_name} @ {len(y_fit):>6,}: "
                f"accuracy {result.accuracy:.4f} ({elapsed:.1f}s)"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "results.npz", **arrays)
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "doi": "10.5281/zenodo.4697906",
                "seed": SEED,
                "digits": DIGITS,
                "n_test": int(len(y_test)),
                "configs": manifest,
            },
            indent=2,
        )
        + "\n"
    )

    size = (args.out / "results.npz").stat().st_size / 1e6
    print(f"Wrote {args.out}/results.npz ({size:.1f} MB) and manifest.json")

    if args.push:
        token = None
        if args.token_file.is_file():
            token = args.token_file.read_text().strip()
        token = token or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise SystemExit(
                f"--push needs a token: {args.token_file} does not exist and "
                "GITHUB_TOKEN is unset. Attach the Renku secret to this "
                "launcher, or pass --token-file."
            )

        digest = oci.push(
            args.push,
            [args.out / "results.npz", args.out / "manifest.json"],
            token=token,
            annotations={
                # Links the package to the repository, which is what makes it
                # inherit the repository's public visibility.
                "org.opencontainers.image.source": (
                    "https://github.com/rokroskar/test-renku-mcp"
                ),
                "org.opencontainers.image.description": (
                    "Frozen MNIST classifier results for the Renku app"
                ),
            },
        )
        print(f"Published {args.push} ({digest})")


if __name__ == "__main__":
    main()
