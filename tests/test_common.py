"""Unit tests for viz/common.py."""
import math

import plotly.graph_objects as go
import torch

from viz import common


def is_empty_message(fig):
    return isinstance(fig, go.Figure) and len(fig.data) == 0 and len(fig.layout.annotations) == 1


def test_to_numpy_replaces_inf_with_nan():
    out = common.to_numpy(torch.tensor([1.0, float("-inf"), 2.0]))
    assert math.isnan(out[1]) and out[0] == 1.0


def test_display_token_and_unique_labels():
    assert common.display_token(" a\n") == "␣a⏎"
    labels = common.unique_labels(["the", " ", "the"])
    assert len(set(labels)) == 3  # repeated tokens stay distinct


def test_heatmap_empty_or_wrong_rank_gives_message():
    assert is_empty_message(common.heatmap(torch.zeros(0, 3)))
    assert is_empty_message(common.heatmap(torch.zeros(3)))


def test_heatmap_basic():
    fig = common.heatmap(torch.randn(3, 4), title="t")
    assert len(fig.data) == 1 and fig.data[0].z.shape == (3, 4)
