# MNIST Digit Classifier Explorer

A Streamlit app that shows how well a simple classifier reads handwritten
digits, and where it goes wrong. Runs as a [Renku](https://renkulab.io) app.

## How the pieces fit

```
Zenodo data connector ──> precompute job (Renku) ──> GHCR OCI artifact ──> app
   read-only mount          fits 9 models             ~3 MB bundle        pulls
```

The app **fits nothing**. It is served to anonymous visitors from several
replicas, and a replica that had to read 130 MB of TSV and fit on one core
would take minutes to answer its first request. So every
classifier/training-size combination is frozen once by the job and published
as an OCI artifact next to the app image; each replica pulls that ~3 MB bundle
at startup.

## Data

The MNIST subset published on Zenodo —
[10.5281/zenodo.4697906](https://doi.org/10.5281/zenodo.4697906), CC-BY-4.0 —
mounted read-only by a Renku data connector: 12,000 training and 6,000 test
images, 28x28 greyscale, as headerless TSV.

> Credit: Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-based learning
> applied to document recognition", *Proceedings of the IEEE* 86(11), 1998.

The mount directory is deployment-specific, so `mnist_data.find_data_dir()`
locates `X_train.tsv` by name under `/home/renku/work`. Set `MNIST_DATA_DIR`
to point somewhere else.

## What the app shows

- **Confusion matrix** — raw counts or row shares, every cell labelled.
- **Per-digit accuracy** — precision, recall and F1 for each digit.
- **Mistakes** — the misclassified test images, filterable by confusion pair.
- **Inspect an image** — any test image with the model's per-digit scores.

## Publishing results

Run inside a Renku job, where the connector is mounted and the buildpack
environment provides scikit-learn:

```bash
/cnb/lifecycle/launcher bash -c '
  cd /home/renku/work/test-renku-mcp &&
  python precompute.py --out /home/renku/work/artifacts \
    --push ghcr.io/rokroskar/test-renku-mcp/model:latest'
```

Pushing needs a GitHub token with `write:packages`, read from
`/secrets/ghcr_token` (a Renku user secret attached to the launcher) or from
`GITHUB_TOKEN`. Nothing else in the pipeline needs a credential: the app pulls
anonymously.

## Running locally

```bash
pip install -r requirements.txt

# either pull what the job published...
streamlit run app.py

# ...or precompute locally first
pip install -r requirements-precompute.txt
MNIST_DATA_DIR=/path/to/record python precompute.py --out artifacts
ARTIFACTS_DIR=artifacts streamlit run app.py
```

`train.py` fits a single configuration and writes `model.joblib` — useful for
poking at one model, but it is not what the app consumes.

## Charts

Chart colours follow the SDSC data-visualization guidelines: the palette is
defined once in `sdsc_plotly_theme.py`, continuous fields use Viridis, and
nothing is encoded by colour alone.
