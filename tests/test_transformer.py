"""Unit tests for the GPT model, loss and generation."""
import math

import pytest
import torch
import torch.nn.functional as F

from llm.transformer import (
    GPT,
    GPTConfig,
    TransformerBlock,
    cross_entropy,
    next_token_distribution,
)


def make(**kw):
    base = dict(vocab_size=20, d_model=16, n_heads=2, n_layers=2, max_seq_len=12)
    base.update(kw)
    return GPT(GPTConfig(**base))


# ----- config --------------------------------------------------------------- #
def test_config_validation():
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=10, d_model=10, n_heads=3)
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=0)
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=10, dropout=1.0)
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=10, positional="rope")
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=10, activation="swish")
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=10, embedding_init="zeros")


def test_config_roundtrip():
    cfg = GPTConfig(vocab_size=30, d_model=32, n_heads=4)
    assert GPTConfig.from_dict(cfg.to_dict()) == cfg


# ----- loss ----------------------------------------------------------------- #
def test_cross_entropy_matches_torch():
    torch.manual_seed(0)
    logits, targets = torch.randn(10, 7) * 5, torch.randint(0, 7, (10,))
    assert torch.allclose(cross_entropy(logits, targets), F.cross_entropy(logits, targets), atol=1e-6)


def test_cross_entropy_stable_for_large_logits():
    loss = cross_entropy(torch.tensor([[1000.0, 0.0]]), torch.tensor([0]))
    assert torch.isfinite(loss) and loss.item() == pytest.approx(0.0, abs=1e-6)


def test_cross_entropy_uniform_is_log_vocab():
    loss = cross_entropy(torch.zeros(4, 9), torch.zeros(4, dtype=torch.long))
    assert loss.item() == pytest.approx(math.log(9), rel=1e-6)


def test_cross_entropy_bad_shapes():
    with pytest.raises(ValueError):
        cross_entropy(torch.zeros(3, 4, 5), torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError):
        cross_entropy(torch.zeros(3, 4), torch.zeros(2, dtype=torch.long))


# ----- model ---------------------------------------------------------------- #
def test_forward_shapes_and_initial_loss():
    torch.manual_seed(0)
    model = make(vocab_size=50)
    ids = torch.randint(0, 50, (4, 10))
    logits, loss = model(ids, ids)
    assert logits.shape == (4, 10, 50)
    assert loss.item() == pytest.approx(math.log(50), abs=0.3)


def test_forward_without_targets_returns_logits_only():
    model = make()
    out = model(torch.randint(0, 20, (2, 5)))
    assert torch.is_tensor(out) and out.shape == (2, 5, 20)


def test_details_structure():
    model = make()
    ids = torch.randint(0, 20, (2, 6))
    logits, details = model(ids, return_details=True)
    assert torch.equal(details["logits"], logits.detach())
    assert len(details["blocks"]) == 2
    b = details["blocks"][0]
    assert b["attention"]["weights"].shape == (2, 2, 6, 6)
    assert b["ffn_hidden"].shape == (2, 6, 64)
    assert torch.allclose(b["after_attn"], b["input"] + b["attn_out"])
    assert torch.allclose(b["output"], b["after_attn"] + b["ffn_out"])
    # with targets: (logits, loss, details)
    out = model(ids, ids, return_details=True)
    assert len(out) == 3


def test_model_is_causal():
    torch.manual_seed(0)
    model = make().eval()
    a = torch.randint(0, 20, (1, 8))
    b = a.clone()
    b[0, 5] = (b[0, 5] + 1) % 20
    la, lb = model(a), model(b)
    assert torch.allclose(la[0, :5], lb[0, :5], atol=1e-6)
    assert not torch.allclose(la[0, 5:], lb[0, 5:])


def test_input_validation():
    model = make()
    with pytest.raises(ValueError):
        model(torch.zeros(13, dtype=torch.long).unsqueeze(0))  # > max_seq_len
    with pytest.raises(ValueError):
        model(torch.zeros(5, dtype=torch.long))  # not 2-D
    with pytest.raises(ValueError):
        model(torch.zeros(1, 0, dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long))
    with pytest.raises(IndexError):
        model(torch.full((1, 3), 99))


def test_weight_tying():
    tied, untied = make(tie_weights=True), make(tie_weights=False)
    assert tied.lm_head.weight is tied.tok_emb.weight
    assert untied.num_parameters() == tied.num_parameters() + 20 * 16


@pytest.mark.parametrize("positional", ["sinusoidal", "learned"])
@pytest.mark.parametrize("tie", [True, False])
def test_parameter_breakdown_sums(positional, tie):
    model = make(positional=positional, tie_weights=tie)
    bd = model.parameter_breakdown()
    assert bd["total"] == model.num_parameters()
    assert bd["output_head"] == (0 if tie else 20 * 16)
    assert (bd["positional"] > 0) == (positional == "learned")


def test_backward_gives_finite_grads_everywhere():
    model = make()
    ids = torch.randint(0, 20, (2, 6))
    _, loss = model(ids, ids)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_block_residual_identity_when_sublayers_zeroed():
    cfg = GPTConfig(vocab_size=5, d_model=8, n_heads=2)
    block = TransformerBlock(cfg)
    for p in list(block.attn.out_proj.parameters()) + list(block.ffn.fc_out.parameters()):
        torch.nn.init.zeros_(p)
    x = torch.randn(1, 4, 8)
    assert torch.allclose(block(x), x)


def test_can_overfit_tiny_batch():
    torch.manual_seed(0)
    model = make(vocab_size=8, d_model=32, n_layers=2)
    ids = torch.randint(0, 8, (4, 8))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        _, loss = model(ids[:, :-1], ids[:, 1:])
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.3


# ----- sampling ------------------------------------------------------------- #
def test_distribution_greedy():
    probs = next_token_distribution(torch.tensor([[1.0, 5.0, 2.0]]), temperature=0)
    assert torch.equal(probs, torch.tensor([[0.0, 1.0, 0.0]]))


def test_distribution_temperature_effect():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    sharp = next_token_distribution(logits, temperature=0.3)
    flat = next_token_distribution(logits, temperature=3.0)
    assert sharp.max() > flat.max()
    assert torch.allclose(sharp.sum(), torch.tensor(1.0)) and torch.allclose(flat.sum(), torch.tensor(1.0))


def test_distribution_top_k():
    probs = next_token_distribution(torch.tensor([[1.0, 4.0, 3.0, 2.0]]), top_k=2)
    assert (probs > 0).sum() == 2 and probs[0, 1] > 0 and probs[0, 2] > 0
    assert torch.allclose(probs.sum(), torch.tensor(1.0))
    # k larger than the vocabulary is harmless
    assert (next_token_distribution(torch.zeros(1, 3), top_k=10) > 0).sum() == 3


def test_distribution_top_p():
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.07, 0.03]]))
    probs = next_token_distribution(logits, top_p=0.85)
    assert (probs > 0).sum() == 2  # 0.6 + 0.3 reaches 0.85
    assert torch.allclose(probs.sum(), torch.tensor(1.0))
    only_best = next_token_distribution(logits, top_p=0.1)
    assert (only_best > 0).sum() == 1  # always keeps at least one token


def test_distribution_invalid_arguments():
    z = torch.zeros(1, 3)
    for kw in ({"temperature": -1}, {"top_k": 0}, {"top_p": 0.0}, {"top_p": 1.5}):
        with pytest.raises(ValueError):
            next_token_distribution(z, **kw)


# ----- generation ----------------------------------------------------------- #
def test_generate_length_range_and_context_cropping():
    model = make(max_seq_len=8)
    start = torch.zeros(2, 3, dtype=torch.long)
    out = model.generate(start, max_new_tokens=15)  # longer than the context window
    assert out.shape == (2, 18)
    assert out.min() >= 0 and out.max() < 20
    assert torch.equal(out[:, :3], start)


def test_generate_is_reproducible_with_seed():
    model = make()
    start = torch.zeros(1, 2, dtype=torch.long)
    a = model.generate(start, 10, generator=torch.Generator().manual_seed(7))
    b = model.generate(start, 10, generator=torch.Generator().manual_seed(7))
    c = model.generate(start, 10, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_generate_greedy_is_deterministic_and_restores_mode():
    model = make(dropout=0.3)
    model.train()
    start = torch.zeros(1, 2, dtype=torch.long)
    assert torch.equal(model.generate(start, 6, temperature=0), model.generate(start, 6, temperature=0))
    assert model.training  # previous mode restored


def test_generate_zero_tokens_and_validation():
    model = make()
    start = torch.zeros(1, 2, dtype=torch.long)
    assert torch.equal(model.generate(start, 0), start)
    with pytest.raises(ValueError):
        model.generate(start, -1)
    with pytest.raises(ValueError):
        model.generate(torch.zeros(1, 0, dtype=torch.long), 3)
