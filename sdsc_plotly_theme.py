"""SDSC plotly theme.

Registers a plotly template named "sdsc" and sets it as the default. Import once
per session.

The colorway is the CVD-safe SDSC categorical palette ordered
most-distinguishable-first (keep to <= 5 series). Continuous fields default to
Viridis (perceptually uniform, greyscale-safe). Never hard-code these hex values
in a plot -- reference the template instead.
"""

import plotly.io as pio
import plotly.graph_objects as go

SDSC_COLORWAY = ["#26235c", "#73a235", "#b34a00", "#56b4e9", "#cc79a7"]

pio.templates["sdsc"] = go.layout.Template(layout={
    "colorway": SDSC_COLORWAY,
    "colorscale": {"sequential": "Viridis"},
    "font": {"size": 12},               # 12px minimum chart text
    "xaxis": {"showgrid": False},
    "yaxis": {"gridcolor": "#e5e5e5"},   # horizontal gridlines only
    "plot_bgcolor": "white",
})

pio.templates.default = "sdsc"
