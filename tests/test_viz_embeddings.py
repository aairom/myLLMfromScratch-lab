"""Unit tests for viz/embeddings.py."""
import plotly.graph_objects as go
import pytest
import torch

from llm.positional import sinusoidal_table
from viz import embeddings as ve


def is_empty_message(fig):
    return isinstance(fig, go.Figure) and len(fig.data) == 0 and len(fig.layout.annotations) == 1


def test_pca_shapes_and_variance():
    torch.manual_seed(0)
    coords, explained = ve.pca_2d(torch.randn(10, 5))
    assert coords.shape == (10, 2) and 0 < explained.sum() <= 1.0 + 1e-9


def test_pca_recovers_dominant_axis():
    x = torch.zeros(20, 3)
    x[:, 0] = torch.linspace(-5, 5, 20)
    coords, explained = ve.pca_2d(x)
    assert explained[0] == pytest.approx(1.0, abs=1e-6)
    assert coords[:, 0].abs().max() == pytest.approx(5.0, abs=1e-5)


def test_pca_degenerate_inputs():
    for m in (torch.ones(1, 4), torch.ones(5, 4), torch.zeros(3, 1)):
        coords, explained = ve.pca_2d(m)
        assert coords.shape == (m.size(0), 2) and torch.isfinite(coords).all() and torch.isfinite(explained).all()


def test_cosine_similarity_matrix():
    m = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]])
    s = ve.cosine_similarity_matrix(m)
    assert torch.allclose(torch.diag(s), torch.ones(4, dtype=torch.double))
    assert s[0, 1] == pytest.approx(1.0) and s[0, 2] == pytest.approx(0.0) and s[0, 3] == pytest.approx(-1.0)
    assert torch.isfinite(ve.cosine_similarity_matrix(torch.zeros(2, 3))).all()  # zero rows are safe


def test_embedding_figures():
    emb = torch.randn(6, 100)
    labels = list("abcdef")
    heat = ve.embedding_heatmap(emb, labels, max_dims=32)
    assert heat.data[0].z.shape == (6, 32)
    assert len(ve.embedding_scatter(emb, labels).data[0].x) == 6
    assert ve.similarity_heatmap(emb, labels).data[0].z.shape == (6, 6)
    for f in (ve.embedding_heatmap(torch.zeros(0, 4), []), ve.embedding_scatter(torch.zeros(0, 4), []),
              ve.similarity_heatmap(torch.zeros(0, 4), [])):
        assert is_empty_message(f)


def test_positional_figures():
    table = sinusoidal_table(16, 8)
    assert ve.positional_heatmap(table).data[0].z.shape == (16, 8)
    curves = ve.positional_curves(table, dims=[0, 1, 99])  # 99 is out of range -> ignored
    assert [t.name for t in curves.data] == ["dim 0 (sin)", "dim 1 (cos)"]
    assert len(ve.positional_curves(table).data) == 6
    assert is_empty_message(ve.positional_curves(table, dims=[99]))
    assert is_empty_message(ve.positional_curves(torch.zeros(0, 4)))
