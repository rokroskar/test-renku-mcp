"""Streamlit app: how well does a simple model read handwritten digits?

Data: MNIST subset published on Zenodo, DOI 10.5281/zenodo.4697906, CC-BY-4.0.
Credit: Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-based learning
applied to document recognition", Proceedings of the IEEE 86(11), 1998.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import precision_recall_fscore_support

import sdsc_plotly_theme  # noqa: F401  -- registers and activates the "sdsc" template
from mnist_data import as_image, find_data_dir, load_split
from mnist_model import DIGITS, MODELS, fit_and_evaluate, subsample

DOI_URL = "https://doi.org/10.5281/zenodo.4697906"

st.set_page_config(
    page_title="MNIST Digit Classifier Explorer",
    page_icon="🔢",
    layout="wide",
)


@st.cache_data(show_spinner="Reading the Zenodo record…")
def load_data():
    data_dir = find_data_dir()
    x_train, y_train = load_split(data_dir, "train")
    x_test, y_test = load_split(data_dir, "test")
    return str(data_dir), x_train, y_train, x_test, y_test


@st.cache_resource(show_spinner="Training…", max_entries=4)
def train(model_name: str, n_train: int, seed: int):
    _, x_train, y_train, x_test, y_test = load_data()
    x_fit, y_fit = subsample(x_train, y_train, n_train, seed)
    return fit_and_evaluate(model_name, x_fit, y_fit, x_test, y_test, seed)


def digit_thumbnail(row: np.ndarray) -> np.ndarray:
    """A 28x28 row as a uint8 greyscale image for st.image."""
    return (as_image(row) * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------

st.sidebar.header("Model")
model_name = st.sidebar.selectbox("Classifier", list(MODELS))
n_train = st.sidebar.select_slider(
    "Training images",
    options=[500, 1000, 2000, 4000, 8000, 12000],
    value=2000,
    help="Larger is more accurate and slower to fit.",
)
seed = st.sidebar.number_input("Random seed", value=0, step=1)

try:
    data_dir, x_train, y_train, x_test, y_test = load_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.caption(f"Record mounted at `{data_dir}`")

result = train(model_name, int(n_train), int(seed))

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("MNIST Digit Classifier Explorer")
st.markdown(
    f"**{result.model_name}** fitted on {result.n_train:,} of {len(y_train):,} "
    f"training images, evaluated on all {len(y_test):,} held-out test images. "
    f"Data: [MNIST subset on Zenodo]({DOI_URL}) (CC-BY-4.0)."
)

top = st.columns(4)
top[0].metric("Test accuracy", f"{result.accuracy:.1%}")
top[1].metric("Misclassified", f"{len(result.errors):,}")
top[2].metric("Training images", f"{result.n_train:,}")
top[3].metric("Test images", f"{len(result.y_true):,}")

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
    counts = result.confusion
    matrix = counts / counts.sum(axis=1, keepdims=True) if normalise else counts

    figure = px.imshow(
        matrix,
        x=DIGITS,
        y=DIGITS,
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
    figure.update_xaxes(tickmode="array", tickvals=DIGITS, side="bottom")
    figure.update_yaxes(tickmode="array", tickvals=DIGITS)
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
    precision, recall, f1, support = precision_recall_fscore_support(
        result.y_true, result.y_pred, labels=DIGITS, zero_division=0
    )
    per_digit = pd.DataFrame(
        {
            "Digit": DIGITS,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Test images": support,
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
    figure.update_xaxes(tickmode="array", tickvals=DIGITS)
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

with mistakes_tab:
    errors = result.errors
    if len(errors) == 0:
        st.success("No misclassified test images.")
    else:
        pair_options = ["Any"] + [
            f"{t} read as {p}"
            for t, p in sorted(
                {(int(result.y_true[i]), int(result.y_pred[i])) for i in errors}
            )
        ]
        chosen = st.selectbox("Confusion", pair_options)
        shown = errors
        if chosen != "Any":
            true_digit, pred_digit = int(chosen[0]), int(chosen[-1])
            shown = np.array(
                [
                    i
                    for i in errors
                    if result.y_true[i] == true_digit
                    and result.y_pred[i] == pred_digit
                ]
            )

        st.write(f"{len(shown):,} misclassified images; showing up to 24.")
        for block in np.array_split(shown[:24], 4):
            if len(block) == 0:
                continue
            for column, index in zip(st.columns(6), block):
                column.image(
                    digit_thumbnail(x_test[index]),
                    caption=f"#{index} · is {result.y_true[index]}, "
                    f"read {result.y_pred[index]}",
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
            np.random.default_rng().integers(len(y_test))
        )
    if controls[2].button("Random mistake", use_container_width=True):
        if len(result.errors):
            st.session_state.pending_index = int(
                np.random.default_rng().choice(result.errors)
            )
        else:
            st.toast("This model got every test image right.")

    index = controls[0].number_input(
        "Test image index",
        min_value=0,
        max_value=len(y_test) - 1,
        value=min(st.session_state.pending_index, len(y_test) - 1),
        step=1,
    )
    st.session_state.pending_index = int(index)

    index = int(index)
    left, right = st.columns([1, 2])
    with left:
        st.image(digit_thumbnail(x_test[index]), width=240)
        truth = int(result.y_true[index])
        prediction = int(result.y_pred[index])
        st.metric("True digit", truth)
        st.metric(
            "Predicted digit",
            prediction,
            delta="correct" if truth == prediction else "wrong",
            delta_color="normal" if truth == prediction else "inverse",
        )

    with right:
        if result.scores is None:
            st.info(
                f"{result.model_name} reports decision margins rather than "
                "calibrated probabilities."
            )
            margins = result.estimator.decision_function(
                x_test[index : index + 1]
            ).ravel()
            frame = pd.DataFrame({"Digit": DIGITS, "Value": margins})
            title, fmt = "Decision margin per digit", None
        else:
            frame = pd.DataFrame(
                {"Digit": DIGITS, "Value": result.scores[index]}
            )
            title, fmt = "Predicted probability per digit", ".0%"

        # One variable, so one colour: hue here would encode nothing.
        figure = px.bar(frame, x="Digit", y="Value", title=title)
        figure.update_xaxes(tickmode="array", tickvals=DIGITS)
        if fmt:
            figure.update_yaxes(range=[0, 1], tickformat=fmt)
        figure.update_layout(height=420, showlegend=False)
        st.plotly_chart(figure, use_container_width=True)
