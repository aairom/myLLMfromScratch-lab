"""Unit tests for attention and multi-head attention."""
import pytest
import torch
import torch.nn.functional as F

from llm.attention import (
    MultiHeadAttention,
    attention_scores,
    causal_mask,
    scaled_dot_product_attention,
)


def test_causal_mask_is_lower_triangular():
    m = causal_mask(4)
    assert m.dtype == torch.bool
    assert torch.equal(m, torch.tril(torch.ones(4, 4, dtype=torch.bool)))


def test_attention_rows_sum_to_one():
    torch.manual_seed(0)
    q = k = v = torch.randn(2, 3, 5, 4)
    _, w = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 5), atol=1e-5)


def test_causal_mask_zeroes_future_weights():
    torch.manual_seed(0)
    q = k = v = torch.randn(1, 6, 8)
    _, w = scaled_dot_product_attention(q, k, v, causal_mask(6))
    assert torch.all(torch.triu(w, diagonal=1) == 0)
    assert torch.allclose(w.sum(-1), torch.ones(1, 6), atol=1e-5)
    assert torch.allclose(w[0, 0], torch.tensor([1.0, 0, 0, 0, 0, 0]))  # first token sees only itself


def test_fully_masked_row_gives_zeros_not_nan():
    q = k = v = torch.randn(1, 3, 4)
    mask = torch.zeros(3, 3, dtype=torch.bool)
    out, w = scaled_dot_product_attention(q, k, v, mask)
    assert torch.isfinite(out).all() and torch.all(w == 0)


def test_scores_scaled_by_sqrt_dk():
    q = k = torch.ones(1, 2, 16)
    assert torch.allclose(attention_scores(q, k), torch.full((1, 2, 2), 16 / 4.0))


def test_matches_torch_reference():
    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 4, 7, 8) for _ in range(3))
    out, _ = scaled_dot_product_attention(q, k, v, causal_mask(7))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)


def test_uniform_attention_averages_values():
    q = k = torch.zeros(1, 4, 2)  # identical scores -> uniform weights
    v = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    out, w = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(w, torch.full((1, 4, 4), 0.25))
    assert torch.allclose(out[0, 0], v[0].mean(0))


# ----- multi-head ----------------------------------------------------------- #
def test_mha_output_shape_and_details():
    mha = MultiHeadAttention(16, 4)
    x = torch.randn(2, 5, 16)
    out, d = mha(x, return_details=True)
    assert out.shape == x.shape
    assert d["q"].shape == (2, 4, 5, 4)
    assert d["weights"].shape == (2, 4, 5, 5)
    assert d["head_output"].shape == (2, 4, 5, 4)
    assert torch.equal(mha(x), out)  # same output without details (eval-equivalent, no dropout)


def test_mha_matches_torch_reference_from_details():
    torch.manual_seed(2)
    mha = MultiHeadAttention(12, 3).eval()
    x = torch.randn(2, 6, 12)
    out, d = mha(x, return_details=True)
    ref_heads = F.scaled_dot_product_attention(d["q"], d["k"], d["v"], is_causal=True)
    assert torch.allclose(d["head_output"], ref_heads, atol=1e-5)
    merged = ref_heads.transpose(1, 2).reshape(2, 6, 12)
    assert torch.allclose(out, mha.out_proj(merged), atol=1e-5)


def test_mha_is_causal():
    torch.manual_seed(3)
    mha = MultiHeadAttention(8, 2).eval()
    x = torch.randn(1, 6, 8)
    x2 = x.clone()
    x2[0, 4:] += 5.0  # change only the future tokens
    assert torch.allclose(mha(x)[0, :4], mha(x2)[0, :4], atol=1e-6)
    assert not torch.allclose(mha(x)[0, 5], mha(x2)[0, 5])


def test_mha_non_causal_sees_future():
    mha = MultiHeadAttention(8, 2, causal=False).eval()
    x = torch.randn(1, 4, 8)
    x2 = x.clone()
    x2[0, 3] += 5.0
    assert not torch.allclose(mha(x)[0, 0], mha(x2)[0, 0])


def test_single_head_equals_full_width():
    mha = MultiHeadAttention(8, 1)
    assert mha.d_k == 8


def test_parameter_count():
    mha = MultiHeadAttention(16, 4)
    assert sum(p.numel() for p in mha.parameters()) == 4 * (16 * 16 + 16)
    nb = MultiHeadAttention(16, 4, bias=False)
    assert sum(p.numel() for p in nb.parameters()) == 4 * 16 * 16


def test_gradients_flow_to_all_projections():
    mha = MultiHeadAttention(8, 2)
    mha(torch.randn(2, 4, 8)).sum().backward()
    for p in mha.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_invalid_configuration_and_input():
    with pytest.raises(ValueError):
        MultiHeadAttention(10, 3)
    with pytest.raises(ValueError):
        MultiHeadAttention(8, 0)
    mha = MultiHeadAttention(8, 2)
    with pytest.raises(ValueError):
        mha(torch.randn(2, 4, 9))
    with pytest.raises(ValueError):
        mha(torch.randn(4, 8))
    with pytest.raises(ValueError):
        mha(torch.randn(1, 0, 8))


def test_dropout_active_only_in_training():
    torch.manual_seed(0)
    mha = MultiHeadAttention(8, 2, dropout=0.5)
    x = torch.randn(1, 5, 8)
    mha.eval()
    assert torch.equal(mha(x), mha(x))
    mha.train()
    assert not torch.equal(mha(x), mha(x))
