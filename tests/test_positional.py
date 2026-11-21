"""Unit tests for positional encodings."""
import math

import pytest
import torch

from llm.positional import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEncoding,
    build_positional,
    sinusoidal_table,
)


def test_sinusoidal_matches_closed_form():
    L, d = 12, 8
    table = sinusoidal_table(L, d)
    for pos in range(L):
        for i in range(d // 2):
            angle = pos / (10000 ** (2 * i / d))
            assert table[pos, 2 * i].item() == pytest.approx(math.sin(angle), abs=1e-5)
            assert table[pos, 2 * i + 1].item() == pytest.approx(math.cos(angle), abs=1e-5)


def test_position_zero_pattern():
    table = sinusoidal_table(4, 6)
    assert torch.allclose(table[0, 0::2], torch.zeros(3))  # sin(0)
    assert torch.allclose(table[0, 1::2], torch.ones(3))  # cos(0)


def test_odd_d_model():
    table = sinusoidal_table(5, 7)
    assert table.shape == (5, 7)
    assert torch.isfinite(table).all()


def test_values_bounded_and_rows_unique():
    table = sinusoidal_table(64, 16)
    assert table.abs().max() <= 1.0
    assert torch.unique(table, dim=0).shape[0] == 64


def test_module_adds_table_and_has_no_parameters():
    pe = SinusoidalPositionalEncoding(8, max_len=16)
    x = torch.zeros(2, 5, 8)
    out = pe(x)
    assert out.shape == x.shape
    assert torch.allclose(out[0], pe.encoding(5))
    assert list(pe.parameters()) == []


def test_sequence_too_long():
    with pytest.raises(ValueError):
        SinusoidalPositionalEncoding(8, 4)(torch.zeros(1, 5, 8))
    with pytest.raises(ValueError):
        LearnedPositionalEmbedding(8, 4)(torch.zeros(1, 5, 8))


def test_learned_is_trainable():
    pe = LearnedPositionalEmbedding(8, max_len=10)
    pe(torch.zeros(2, 3, 8)).sum().backward()
    assert pe.weight.grad[:3].abs().sum() > 0
    assert pe.weight.grad[3:].abs().sum() == 0


def test_invalid_sizes():
    with pytest.raises(ValueError):
        sinusoidal_table(0, 4)
    with pytest.raises(ValueError):
        LearnedPositionalEmbedding(0, 4)


def test_factory():
    assert isinstance(build_positional("sinusoidal", 8, 16), SinusoidalPositionalEncoding)
    assert isinstance(build_positional("learned", 8, 16), LearnedPositionalEmbedding)
    with pytest.raises(ValueError):
        build_positional("rope", 8, 16)
