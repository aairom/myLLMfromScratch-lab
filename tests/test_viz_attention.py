"""Unit tests for viz/attention.py."""
import math

import plotly.graph_objects as go
import torch

from llm.attention import MultiHeadAttention
from llm.layers import LayerNorm
from viz import attention as va


def is_empty_message(fig):
    return isinstance(fig, go.Figure) and len(fig.data) == 0 and len(fig.layout.annotations) == 1


def attention_inputs(t=5, heads=2):
    d_model = 4 * heads  # keeps d_k = 4 and d_model divisible by heads
    mha = MultiHeadAttention(d_model, heads).eval()
    _, d = mha(torch.randn(1, t, d_model), return_details=True)
    return d, ["the", " cat", " sat", " on", " the"][:t]


def test_attention_heatmap_masked_cells_are_blank():
    d, toks = attention_inputs()
    probs = va.attention_heatmap(d["weights"][0, 0], toks)
    assert probs.data[0].z.shape == (5, 5)
    scores = va.attention_heatmap(d["scores"][0, 0], toks, pre_softmax=True)
    z = scores.data[0].z
    assert math.isnan(z[0][4]) and not math.isnan(z[4][0])  # future cell is masked


def test_heads_grid_layout():
    d, toks = attention_inputs(heads=3)
    fig = va.heads_grid(d["weights"][0], toks, max_cols=2)
    assert len(fig.data) == 3
    assert is_empty_message(va.heads_grid(torch.zeros(2, 2), toks))


def test_qkv_and_received_figures():
    d, toks = attention_inputs()
    assert va.qkv_heatmap(d["q"][0, 1], toks, "Query", head=1).data[0].z.shape == (5, 4)
    received = va.attention_received_figure(d["weights"][0], toks)  # averaged over heads
    assert len(received.data[0].y) == 5
    assert is_empty_message(va.attention_received_figure(torch.zeros(0, 0), []))


def test_layernorm_figures():
    x = torch.randn(4, 16) * 5 + 3
    out = LayerNorm(16)(x)
    dist = va.layernorm_distribution(x, out)
    assert len(dist.data) == 2
    stats = va.layernorm_token_stats(x, out, list("abcd"))
    assert len(stats.data) == 4
    assert is_empty_message(va.layernorm_distribution(torch.zeros(0), torch.zeros(0)))
    assert is_empty_message(va.layernorm_token_stats(x, out[:2], list("abcd")))
