"""Step 1: turn the Zenodo TSVs into a compact dataset bundle.

Reads the data connector mounted in the job and writes uint8 arrays. This is
the slow part -- parsing 130 MB of text -- and doing it once means steps 2 and
3 start from a few MB instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import publish
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mnist_data import find_data_dir, load_split  # noqa: E402

SOURCE = "https://github.com/rokroskar/test-renku-mcp"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--push", default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or find_data_dir()
    print(f"Reading the Zenodo record from {data_dir}")

    arrays = {}
    counts = {}
    for split in ("train", "test"):
        features, labels = load_split(data_dir, split)
        # load_split scales to [0, 1]; store the original integer intensities.
        arrays[f"x_{split}"] = np.rint(features * 255).astype(np.uint8)
        arrays[f"y_{split}"] = labels.astype(np.uint8)
        counts[split] = int(len(labels))
        print(f"  {split}: {len(labels):,} images")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "dataset.npz", **arrays)
    (args.out / "dataset.json").write_text(
        json.dumps(
            {
                "doi": "10.5281/zenodo.4697906",
                "counts": counts,
                "pixels": 784,
                "dtype": "uint8",
            },
            indent=2,
        )
        + "\n"
    )

    size = (args.out / "dataset.npz").stat().st_size / 1e6
    print(f"Wrote {args.out}/dataset.npz ({size:.1f} MB)")
    publish(args.push, args.out, ["dataset.npz", "dataset.json"], SOURCE)


if __name__ == "__main__":
    main()
