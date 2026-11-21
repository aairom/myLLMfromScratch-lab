"""Figures for embeddings and positional encodings."""
from __future__ import annotations

from typing import Optional, Sequence

import plotly.graph_objects as go
import torch

from .common import empty_figure, heatmap, to_numpy


def pca_2d(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project rows of ``matrix`` (N, D) onto their first two principal components.

    Returns ``(coords (N, 2), explained_variance_ratio (2,))``. Degenerate inputs
    (one row, one column, or all rows identical) give zeros instead of failing.
    """
    x = matrix.detach().cpu().double()
    n, d = x.shape
    coords = torch.zeros(n, 2, dtype=torch.double)
    explained = torch.zeros(2, dtype=torch.double)
    if n < 2 or d < 1:
        return coords, explained
    centred = x - x.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(centred, full_matrices=False)
    k = min(2, vh.size(0))
    coords[:, :k] = centred @ vh[:k].T
    total = float((s**2).sum())
    if total > 0:
        explained[:k] = (s[:k] ** 2) / total
    return coords, explained


def cosine_similarity_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity of the rows: ``cos(a, b) = a.b / (|a||b|)``."""
    x = matrix.detach().cpu().double()
    norm = x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    unit = x / norm
    return unit @ unit.T


def embedding_heatmap(vectors: torch.Tensor, labels: Sequence[str], max_dims: int = 64) -> go.Figure:
    """Rows = tokens, columns = embedding dimensions (first ``max_dims`` shown)."""
    v = vectors.detach().cpu()
    if v.dim() != 2 or v.numel() == 0:
        return empty_figure("no embeddings to display")
    shown = v[:, :max_dims]
    return heatmap(
        shown,
        x_labels=[str(i) for i in range(shown.size(1))],
        y_labels=list(labels),
        title=f"Embedding vectors ({shown.size(0)} tokens x {shown.size(1)} of {v.size(1)} dims)",
        x_title="dimension",
        colorbar_title="value",
    )


def embedding_scatter(matrix: torch.Tensor, labels: Sequence[str]) -> go.Figure:
    """2-D PCA projection of the embedding vectors, one labelled point per token."""
    if matrix.dim() != 2 or matrix.size(0) == 0:
        return empty_figure("no embeddings to display")
    coords, explained = pca_2d(matrix)
    fig = go.Figure(
        go.Scatter(
            x=coords[:, 0].tolist(),
            y=coords[:, 1].tolist(),
            mode="markers+text",
            text=list(labels),
            textposition="top center",
        )
    )
    fig.update_layout(
        title=f"PCA of embeddings (explains {100 * float(explained.sum()):.0f}% of the variance)",
        xaxis_title=f"PC1 ({100 * float(explained[0]):.0f}%)",
        yaxis_title=f"PC2 ({100 * float(explained[1]):.0f}%)",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def similarity_heatmap(matrix: torch.Tensor, labels: Sequence[str]) -> go.Figure:
    """Cosine similarity between every pair of tokens (1 = same direction)."""
    if matrix.dim() != 2 or matrix.size(0) == 0:
        return empty_figure("no embeddings to display")
    return heatmap(
        cosine_similarity_matrix(matrix),
        x_labels=list(labels),
        y_labels=list(labels),
        title="Cosine similarity between token embeddings",
        zmin=-1.0,
        zmax=1.0,
        colorbar_title="cosine",
    )


def positional_heatmap(table: torch.Tensor, title: str = "Positional encoding") -> go.Figure:
    """Rows = positions, columns = dimensions."""
    return heatmap(table, title=title, x_title="dimension", y_title="position", zmin=-1.0, zmax=1.0, colorbar_title="value")


def positional_curves(table: torch.Tensor, dims: Optional[Sequence[int]] = None) -> go.Figure:
    """Value of selected dimensions as a function of position (the sine/cosine waves)."""
    arr = to_numpy(table)
    if arr.ndim != 2 or arr.size == 0:
        return empty_figure("no positional table")
    chosen = [d for d in (dims if dims is not None else range(min(6, arr.shape[1]))) if 0 <= d < arr.shape[1]]
    if not chosen:
        return empty_figure("no valid dimensions selected")
    fig = go.Figure()
    for d in chosen:
        kind = "sin" if d % 2 == 0 else "cos"
        fig.add_trace(go.Scatter(x=list(range(arr.shape[0])), y=arr[:, d], mode="lines", name=f"dim {d} ({kind})"))
    fig.update_layout(
        title="Positional encoding: one wave per dimension",
        xaxis_title="position",
        yaxis_title="value",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
