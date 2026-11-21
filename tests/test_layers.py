"""Unit tests for LayerNorm, GELU and the feed-forward network."""
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from llm.layers import ACTIVATIONS, FeedForward, LayerNorm, gelu, relu


def test_layernorm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 16) * 4 + 2
    ours, ref = LayerNorm(16), nn.LayerNorm(16)
    with torch.no_grad():
        ours.weight.copy_(torch.randn(16))
        ours.bias.copy_(torch.randn(16))
        ref.weight.copy_(ours.weight)
        ref.bias.copy_(ours.bias)
    assert torch.allclose(ours(x), ref(x), atol=1e-5)


def test_layernorm_statistics():
    x = torch.randn(4, 7, 32) * 10 + 5
    s = LayerNorm(32).stages(x)
    assert torch.allclose(s["normalized"].mean(-1), torch.zeros(4, 7), atol=1e-5)
    assert torch.allclose(s["normalized"].var(-1, unbiased=False), torch.ones(4, 7), atol=1e-3)
    assert s["mean"].shape == (4, 7, 1) and s["var"].shape == (4, 7, 1)


def test_layernorm_constant_input_is_finite():
    out = LayerNorm(8)(torch.full((2, 3, 8), 3.0))
    assert torch.isfinite(out).all() and torch.allclose(out, torch.zeros_like(out))


def test_layernorm_gradients_and_validation():
    ln = LayerNorm(8)
    ln(torch.randn(2, 8)).sum().backward()
    assert ln.weight.grad is not None and ln.bias.grad is not None
    with pytest.raises(ValueError):
        ln(torch.randn(2, 9))
    with pytest.raises(ValueError):
        LayerNorm(0)


def test_gelu_matches_torch_tanh_variant():
    x = torch.linspace(-6, 6, 101)
    assert torch.allclose(gelu(x), F.gelu(x, approximate="tanh"), atol=1e-5)
    assert gelu(torch.tensor(0.0)) == 0


def test_relu():
    assert torch.equal(relu(torch.tensor([-1.0, 0.0, 2.0])), torch.tensor([0.0, 0.0, 2.0]))


def test_ffn_shapes_and_default_width():
    ffn = FeedForward(16)
    assert ffn.d_hidden == 64
    x = torch.randn(2, 5, 16)
    out, hidden = ffn(x, return_hidden=True)
    assert out.shape == x.shape and hidden.shape == (2, 5, 64)


def test_ffn_is_position_wise():
    """Each token is transformed independently of the others."""
    ffn = FeedForward(8).eval()
    x = torch.randn(1, 4, 8)
    assert torch.allclose(ffn(x)[0, 2], ffn(x[:, 2:3])[0, 0], atol=1e-6)


def test_ffn_parameter_count_and_options():
    ffn = FeedForward(8, d_hidden=20)
    assert sum(p.numel() for p in ffn.parameters()) == 8 * 20 + 20 + 20 * 8 + 8
    assert set(ACTIVATIONS) == {"gelu", "relu"}
    FeedForward(8, activation="relu")(torch.randn(1, 2, 8))


def test_ffn_invalid_arguments():
    with pytest.raises(ValueError):
        FeedForward(8, activation="swish")
    with pytest.raises(ValueError):
        FeedForward(0)
    with pytest.raises(ValueError):
        FeedForward(8, d_hidden=0)
