"""Streamlit app: how well does a simple model read handwritten digits?

Data: MNIST subset published on Zenodo, DOI 10.5281/zenodo.4697906, CC-BY-4.0.
Credit: Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-based learning
applied to document recognition", Proceedings of the IEEE 86(11), 1998.

The app fits nothing. Every classifier/training-size combination is frozen by
the precompute job in Renku and published to the GitHub container registry;
the app pulls that ~3 MB bundle at startup. It is served from several
replicas, and fitting per pod would make each cold start cost minutes.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import oci
import sdsc_plotly_theme  # noqa: F401  -- registers and activates the "sdsc" template
from mnist_artifacts import Artifacts, find_artifacts_dir, per_digit_metrics

DOI_URL = "https://doi.org/10.5281/zenodo.4697906"
RESULTS_REFERENCE = os.environ.get(
    "RESULTS_REFERENCE", "ghcr.io/rokroskar/test-renku-mcp/model:latest"
)

st.set_page_config(
    page_title="MNIST Digit Classifier Explorer",
    page_icon="\U0001F522",
    layout="wide",
)


# cache_resource, not cache_data: the arrays are never mutated, so every
# session should share one copy rather than be handed a fresh deserialisation.
@st.cache_resource(show_spinner="Fetching the published results…")
def load_artifacts() -> tuple[Artifacts, str]:
    """Prefer a local copy; otherwise pull the bundle the job published."""
    try:
        return Artifacts(find_artifacts_dir()), "local files"
    except FileNotFoundError:
        directory = Path(tempfile.mkdtemp(prefix="mnist-results-"))
        oci.pull(RESULTS_REFERENCE, directory)
        return Artifacts(directory), RESULTS_REFERENCE


try:
    artifacts, source = load_artifacts()
except (FileNotFoundError, oci.RegistryError) as exc:
    st.error(
        f"No precomputed results available: {exc}\n\n"
        "Run the *Train the classifier* job in the Renku project to publish "
        "them."
    )
    st.stop()

digits = artifacts.digits

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------

st.sidebar.header("Model")
model_name = st.sidebar.selectbox("Classifier", artifacts.models)
n_train = st.sidebar.select_slider(
    "Training images",
    options=artifacts.sizes_for(model_name),
    value=artifacts.sizes_for(model_name)[-1],
)
config = artifacts.find(model_name, int(n_train))

predictions = artifacts.predictions(config)
labels = artifacts.labels
errors = np.flatnonzero(labels != predictions)

st.sidebar.caption(
    f"Fitted once by the Renku job in {config.fit_seconds:.0f}s, seed "
    f"{artifacts.seed}. The app itself trains nothing.\n\n"
    f"Results from `{source}`."
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("MNIST Digit Classifier Explorer")
st.markdown(
    f"**{config.model}** fitted on {config.n_train:,} training images and "
    f"evaluated on all {artifacts.n_test:,} held-out test images. "
    f"Data: [MNIST subset on Zenodo]({DOI_URL}) (CC-BY-4.0)."
)

top = st.columns(4)
top[0].metric("Test accuracy", f"{config.accuracy:.1%}")
top[1].metric("Misclassified", f"{len(errors):,}")
top[2].metric("Training images", f"{config.n_train:,}")
top[3].metric("Test images", f"{artifacts.n_test:,}")

overview_tab, per_digit_tab, mistakes_tab, inspect_tab = st.tabs(
    ["Confusion matrix", "Per-digit accuracy", "Mistakes", "Inspect an image"]
)

# --------------------------------------------------------------------------
# Confusion matrix
# --------------------------------------------------------------------------

with overview_tab:
    normalise = st.checkbox(
        "Show each row as a share of that digit's images",
        value=True,
        help="Row-normalising exposes the off-diagonal confusions that raw "
        "counts hide behind the diagonal.",
    )
    counts = config.confusion
    matrix = counts / counts.sum(axis=1, keepdims=True) if normalise else counts

    figure = px.imshow(
        matrix,
        x=digits,
        y=digits,
        labels={"x": "Predicted digit", "y": "True digit", "color": ""},
        aspect="equal",
        origin="upper",
    )
    # Counts are printed in every cell so the reading never depends on colour.
    figure.update_traces(
        text=counts,
        texttemplate="%{text}",
        hovertemplate="true %{y} → predicted %{x}: %{text} images<extra></extra>",
    )
    figure.update_xaxes(tickmode="array", tickvals=digits, side="bottom")
    figure.update_yaxes(tickmode="array", tickvals=digits)
    figure.update_layout(height=620, coloraxis_showscale=False)
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Cell labels are image counts. Colour "
        + ("encodes the row share." if normalise else "encodes the count.")
    )

# --------------------------------------------------------------------------
# Per-digit metrics
# --------------------------------------------------------------------------

with per_digit_tab:
    metrics = per_digit_metrics(config.confusion)
    per_digit = pd.DataFrame(
        {
            "Digit": digits,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "Test images": metrics["support"],
        }
    )

    long = per_digit.melt(
        id_vars="Digit",
        value_vars=["Precision", "Recall", "F1"],
        var_name="Metric",
        value_name="Score",
    )
    figure = px.bar(
        long,
        x="Digit",
        y="Score",
        color="Metric",
        barmode="group",
        pattern_shape="Metric",  # readable without colour
    )
    figure.update_xaxes(tickmode="array", tickvals=digits)
    figure.update_yaxes(range=[0, 1], tickformat=".0%")
    figure.update_layout(height=440)
    st.plotly_chart(figure, use_container_width=True)

    st.dataframe(
        per_digit.style.format(
            {"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Precision: of the images called *d*, the share that really were *d*. "
        "Recall: of the images that were *d*, the share the model found."
    )

# --------------------------------------------------------------------------
# Mistakes
# --------------------------------------------------------------------------


def thumbnail(index: int) -> np.ndarray:
    return artifacts.images[index].reshape(28, 28)


with mistakes_tab:
    if len(errors) == 0:
        st.success("No misclassified test images.")
    else:
        pairs = sorted({(int(labels[i]), int(predictions[i])) for i in errors})
        options = ["Any"] + [f"{t} read as {p}" for t, p in pairs]
        chosen = st.selectbox("Confusion", options)

        shown = errors
        if chosen != "Any":
            true_digit, pred_digit = int(chosen[0]), int(chosen[-1])
            shown = errors[
                (labels[errors] == true_digit)
                & (predictions[errors] == pred_digit)
            ]

        st.write(f"{len(shown):,} misclassified images; showing up to 24.")
        for block in np.array_split(shown[:24], 4):
            if len(block) == 0:
                continue
            for column, index in zip(st.columns(6), block):
                column.image(
                    thumbnail(index),
                    caption=f"#{index} · is {labels[index]}, "
                    f"read {predictions[index]}",
                    width=90,
                )

# --------------------------------------------------------------------------
# Single-image inspector
# --------------------------------------------------------------------------

with inspect_tab:
    # The buttons are rendered before the number_input so that they can move
    # the shared index: Streamlit forbids writing a widget's state after the
    # widget has been instantiated in the same run.
    if "pending_index" not in st.session_state:
        st.session_state.pending_index = 0

    controls = st.columns([2, 1, 1])
    if controls[1].button("Random image", use_container_width=True):
        st.session_state.pending_index = int(
            np.random.default_rng().integers(artifacts.n_test)
        )
    if controls[2].button("Random mistake", use_container_width=True):
        if len(errors):
            st.session_state.pending_index = int(
                np.random.default_rng().choice(errors)
            )
        else:
            st.toast("This model got every test image right.")

    index = controls[0].number_input(
        "Test image index",
        min_value=0,
        max_value=artifacts.n_test - 1,
        value=min(st.session_state.pending_index, artifacts.n_test - 1),
        step=1,
    )
    st.session_state.pending_index = int(index)
    index = int(index)

    left, right = st.columns([1, 2])
    with left:
        st.image(thumbnail(index), width=240)
        truth = int(labels[index])
        prediction = int(predictions[index])
        st.metric("True digit", truth)
        st.metric(
            "Predicted digit",
            prediction,
            delta="correct" if truth == prediction else "wrong",
            delta_color="normal" if truth == prediction else "inverse",
        )

    with right:
        values = artifacts.scores(config)[index]
        if config.scores_are_probabilities:
            title, axis_format, axis_range = (
                "Predicted probability per digit",
                ".0%",
                [0, 1],
            )
        else:
            st.info(
                f"{config.model} reports decision margins rather than "
                "calibrated probabilities."
            )
            title, axis_format, axis_range = (
                "Decision margin per digit",
                None,
                None,
            )

        # One variable, so one colour: hue here would encode nothing.
        frame = pd.DataFrame({"Digit": digits, "Value": values})
        figure = px.bar(frame, x="Digit", y="Value", title=title)
        figure.update_xaxes(tickmode="array", tickvals=digits)
        if axis_format:
            figure.update_yaxes(range=axis_range, tickformat=axis_format)
        figure.update_layout(height=420, showlegend=False)
        st.plotly_chart(figure, use_container_width=True)
