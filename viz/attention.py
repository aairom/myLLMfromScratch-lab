"""Figures for attention, multi-head attention and layer normalisation."""
from __future__ import annotations

from typing import Optional, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import empty_figure, heatmap, to_numpy, unique_labels


def attention_heatmap(
    weights,
    tokens: Sequence[str],
    title: str = "Attention weights",
    pre_softmax: bool = False,
) -> go.Figure:
    """Attention matrix ``(T, T)``: row = query token, column = key token it looks at.

    With ``pre_softmax=True`` the raw scores are shown (masked cells are blank);
    otherwise rows are probabilities that sum to 1.
    """
    labels = unique_labels(tokens)
    return heatmap(
        weights,
        x_labels=labels,
        y_labels=labels,
        title=title,
        colorscale="RdBu" if pre_softmax else "Blues",
        zmid=0.0 if pre_softmax else None,
        zmin=None if pre_softmax else 0.0,
        zmax=None if pre_softmax else 1.0,
        x_title="key (attended to)",
        y_title="query (looking)",
        colorbar_title="score" if pre_softmax else "weight",
    )


def heads_grid(weights, tokens: Sequence[str], max_cols: int = 4) -> go.Figure:
    """Small multiples: the attention pattern of every head ``weights`` ``(H, T, T)``."""
    arr = to_numpy(weights)
    if arr.ndim != 3 or arr.size == 0:
        return empty_figure("no attention weights")
    n_heads = arr.shape[0]
    cols = min(max_cols, n_heads)
    rows = -(-n_heads // cols)  # ceil division
    labels = unique_labels(tokens)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"head {h}" for h in range(n_heads)])
    for h in range(n_heads):
        fig.add_trace(
            go.Heatmap(z=arr[h], x=labels, y=labels, colorscale="Blues", zmin=0.0, zmax=1.0, showscale=h == 0),
            row=h // cols + 1,
            col=h % cols + 1,
        )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(title="Attention pattern per head", height=260 * rows + 60, margin=dict(l=20, r=20, t=80, b=20))
    return fig


def qkv_heatmap(matrix, tokens: Sequence[str], name: str, head: Optional[int] = None) -> go.Figure:
    """Query / key / value matrix ``(T, d_k)`` of one head: rows = tokens."""
    suffix = f" (head {head})" if head is not None else ""
    return heatmap(
        matrix,
        y_labels=unique_labels(tokens),
        x_labels=[str(i) for i in range(to_numpy(matrix).shape[-1])] if to_numpy(matrix).ndim == 2 else None,
        title=f"{name}{suffix}",
        x_title="dimension within the head",
        colorbar_title="value",
    )


def attention_received_figure(weights, tokens: Sequence[str]) -> go.Figure:
    """How much attention each token *receives*, averaged over queries (and heads if given)."""
    arr = to_numpy(weights)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    if arr.ndim != 2 or arr.size == 0:
        return empty_figure("no attention weights")
    received = arr.mean(axis=0)
    fig = go.Figure(go.Bar(x=unique_labels(tokens), y=received))
    fig.update_layout(
        title="Average attention received by each token",
        xaxis=dict(title="token", type="category"),
        yaxis_title="mean weight",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def layernorm_distribution(before, after, title: str = "LayerNorm: activations before and after") -> go.Figure:
    """Overlaid histograms of the activations before and after normalisation."""
    b, a = to_numpy(before).ravel(), to_numpy(after).ravel()
    if b.size == 0 or a.size == 0:
        return empty_figure("no activations")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=b, name="before", opacity=0.6, nbinsx=40))
    fig.add_trace(go.Histogram(x=a, name="after", opacity=0.6, nbinsx=40))
    fig.update_layout(
        barmode="overlay",
        title=title,
        xaxis_title="activation value",
        yaxis_title="count",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def layernorm_token_stats(before, after, tokens: Sequence[str]) -> go.Figure:
    """Per-token mean and standard deviation of the feature vector, before vs. after."""
    b, a = to_numpy(before), to_numpy(after)
    if b.ndim != 2 or b.shape != a.shape or b.size == 0:
        return empty_figure("expected two matrices of shape (tokens, features)")
    labels = unique_labels(tokens)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("mean per token", "std per token"))
    for name, arr in (("before", b), ("after", a)):
        fig.add_trace(go.Bar(x=labels, y=arr.mean(axis=1), name=name, legendgroup=name), row=1, col=1)
        fig.add_trace(go.Bar(x=labels, y=arr.std(axis=1), name=name, legendgroup=name, showlegend=False), row=1, col=2)
    fig.update_xaxes(type="category")
    fig.update_layout(barmode="group", margin=dict(l=40, r=20, t=60, b=40))
    return fig
