# MNIST Digit Classifier Explorer

A Streamlit app that shows how well a simple classifier reads handwritten
digits, and where it goes wrong. Built to run as a
[Renku](https://renkulab.io) app.

## Data

The MNIST subset published on Zenodo —
[10.5281/zenodo.4697906](https://doi.org/10.5281/zenodo.4697906), CC-BY-4.0 —
mounted read-only by a Renku data connector: 12,000 training and 6,000 test
images, 28x28 greyscale, as headerless TSV.

> Credit: Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-based learning
> applied to document recognition", *Proceedings of the IEEE* 86(11), 1998.

The mount directory is deployment-specific, so `mnist_data.find_data_dir()`
locates `X_train.tsv` by name under `/home/renku/work`. Set `MNIST_DATA_DIR` to
point somewhere else.

## What the app shows

- **Confusion matrix** — raw counts or row shares, every cell labelled.
- **Per-digit accuracy** — precision, recall and F1 for each digit.
- **Mistakes** — the misclassified test images, filterable by confusion pair.
- **Inspect an image** — any test image with the model's per-digit scores.

Classifier, training-set size and seed are set in the sidebar; models are
trained on demand and cached.

## Run it locally

```bash
pip install -r requirements.txt
MNIST_DATA_DIR=/path/to/record streamlit run app.py
```

## Train from the command line

Also runnable as a Renku job:

```bash
python train.py --model "Logistic regression" --n-train 12000 --out models
```

Writes `models/model.joblib` and `models/metrics.json`.

## Charts

Chart colours follow the SDSC data-visualization guidelines: the palette is
defined once in `sdsc_plotly_theme.py`, continuous fields use Viridis, and
nothing is encoded by colour alone.
