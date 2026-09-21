"""Loading the MNIST subset published on Zenodo (DOI 10.5281/zenodo.4697906).

The Renku data connector mounts the record read-only inside the session or app.
The mount directory is deployment-specific, so the files are located by name
rather than by a hard-coded path. Set MNIST_DATA_DIR to override the search.

Record layout (no headers in any file):
    X_train.tsv  12000 x 784 float pixel intensities, 0-255
    y_train.tsv  12000 integer labels, one per line
    X_test.tsv    6000 x 784
    y_test.tsv    6000 labels
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

FILES = ("X_train.tsv", "y_train.tsv", "X_test.tsv", "y_test.tsv")

SEARCH_ROOTS = ("/home/renku/work", ".", "data")


def find_data_dir() -> Path:
    """Return the directory holding the four record files."""
    override = os.environ.get("MNIST_DATA_DIR")
    if override:
        candidate = Path(override)
        if (candidate / "X_train.tsv").is_file():
            return candidate
        raise FileNotFoundError(f"MNIST_DATA_DIR={override} has no X_train.tsv")

    for root in SEARCH_ROOTS:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for match in sorted(root_path.glob("**/X_train.tsv")):
            return match.parent

    raise FileNotFoundError(
        "Could not locate X_train.tsv. Attach the Zenodo data connector "
        "(DOI 10.5281/zenodo.4697906) or set MNIST_DATA_DIR."
    )


def _read_matrix(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t", header=None, dtype=np.float32)
    return frame.to_numpy()


def _read_labels(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t", header=None, dtype=np.int16)
    return frame.to_numpy().ravel()


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one split, with pixels scaled to [0, 1]."""
    features = _read_matrix(data_dir / f"X_{split}.tsv") / 255.0
    labels = _read_labels(data_dir / f"y_{split}.tsv")
    if len(features) != len(labels):
        raise ValueError(
            f"{split}: {len(features)} images but {len(labels)} labels"
        )
    return features, labels


def as_image(row: np.ndarray) -> np.ndarray:
    """Reshape one flattened row back to a 28x28 image."""
    return row.reshape(28, 28)
