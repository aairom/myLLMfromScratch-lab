"""Unit tests for token embeddings."""
import pytest
import torch

from llm.embeddings import INIT_STRATEGIES, TokenEmbedding, init_embedding_


def test_output_shape_and_lookup():
    emb = TokenEmbedding(10, 8)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = emb(ids)
    assert out.shape == (2, 3, 8)
    assert torch.equal(out[0, 1], emb.weight[2])  # row 2 of the table


def test_scale_option():
    emb = TokenEmbedding(5, 16, scale=True)
    ids = torch.tensor([3])
    assert torch.allclose(emb(ids)[0], emb.weight[3] * 4.0)  # sqrt(16) = 4


def test_gradient_only_reaches_used_rows():
    emb = TokenEmbedding(6, 4)
    emb(torch.tensor([1, 1, 3])).sum().backward()
    grad = emb.weight.grad
    assert grad[1].abs().sum() > 0 and grad[3].abs().sum() > 0
    assert grad[0].abs().sum() == 0 and grad[5].abs().sum() == 0
    assert torch.allclose(grad[1], torch.full((4,), 2.0))  # looked up twice


@pytest.mark.parametrize("strategy", INIT_STRATEGIES)
def test_init_statistics(strategy):
    torch.manual_seed(0)
    w = init_embedding_(torch.empty(2000, 64), strategy, std=0.02)
    assert abs(w.mean().item()) < 1e-3
    if strategy == "xavier":
        assert w.std().item() == pytest.approx((2.0 / (2000 + 64)) ** 0.5, rel=0.05)
    else:
        assert w.std().item() == pytest.approx(0.02, rel=0.05)


def test_invalid_init_and_sizes():
    with pytest.raises(ValueError):
        init_embedding_(torch.empty(2, 2), "bogus")
    with pytest.raises(ValueError):
        TokenEmbedding(0, 4)
    with pytest.raises(ValueError):
        TokenEmbedding(4, 0)


def test_out_of_range_ids():
    emb = TokenEmbedding(5, 4)
    with pytest.raises(IndexError):
        emb(torch.tensor([5]))
    with pytest.raises(IndexError):
        emb(torch.tensor([-1]))


def test_empty_input():
    emb = TokenEmbedding(5, 4)
    assert emb(torch.empty(0, dtype=torch.long)).shape == (0, 4)
