"""Shared helpers for the figure builders (no Streamlit, no model logic)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go
import torch

# Characters that are invisible in a plot axis get a visible stand-in.
_VISIBLE = {" ": "␣", "\n": "⏎", "\t": "⇥", "\r": "␍"}


def to_numpy(x) -> np.ndarray:
    """Tensor / list -> float64 NumPy array; non-finite values (e.g. -inf masks) become NaN."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().double().numpy()
    arr = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(arr), arr, np.nan)


def display_token(token: str) -> str:
    """Make whitespace inside a token visible (``" the"`` -> ``"␣the"``)."""
    return "".join(_VISIBLE.get(c, c) for c in token)


def unique_labels(tokens: Sequence[str]) -> list[str]:
    """Axis labels that stay distinct even when a token repeats: ``"0·the"``, ``"4·the"``.

    Plotly merges categorical labels that are equal, which would silently
    collapse repeated tokens into one row/column.
    """
    return [f"{i}·{display_token(t)}" for i, t in enumerate(tokens)]


def empty_figure(message: str, height: int = 300) -> go.Figure:
    """A blank figure carrying only a centred message (used for 'no data yet')."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def heatmap(
    z,
    x_labels: Optional[Sequence[str]] = None,
    y_labels: Optional[Sequence[str]] = None,
    title: str = "",
    colorscale: str = "RdBu",
    zmid: Optional[float] = 0.0,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    x_title: str = "",
    y_title: str = "",
    height: Optional[int] = None,
    colorbar_title: str = "",
) -> go.Figure:
    """Generic 2-D heat map with the row 0 at the top (matrix orientation)."""
    arr = to_numpy(z)
    if arr.ndim != 2 or arr.size == 0:
        return empty_figure(title or "nothing to display")
    fig = go.Figure(
        go.Heatmap(
            z=arr,
            x=list(x_labels) if x_labels is not None else None,
            y=list(y_labels) if y_labels is not None else None,
            colorscale=colorscale,
            zmid=zmid,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=colorbar_title),
            hoverongaps=False,
        )
    )
    fig.update_yaxes(autorange="reversed", title=y_title)
    fig.update_xaxes(title=x_title)
    fig.update_layout(title=title, height=height, margin=dict(l=40, r=20, t=50, b=40))
    return fig
