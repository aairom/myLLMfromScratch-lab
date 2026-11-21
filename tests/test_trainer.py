"""Unit tests for the trainer and gradient clipping."""
import math

import pytest
import torch
from torch import nn

from llm.tokenizer import CharTokenizer
from llm.transformer import GPT, GPTConfig
from training.data import build_corpus, encode_corpus, train_val_split
from training.trainer import TrainConfig, clip_grad_norm, estimate_loss, train

CPU = torch.device("cpu")


def tiny_setup(vocab=12, **cfg_kw):
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=vocab, d_model=16, n_heads=2, n_layers=1, max_seq_len=16, **cfg_kw))
    data = torch.randint(0, vocab, (400,))
    return model, data[:350], data[350:]


# ----- config ---------------------------------------------------------------- #
def test_train_config_validation_and_roundtrip():
    cfg = TrainConfig(steps=5)
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
    for bad in (dict(steps=0), dict(batch_size=0), dict(optimizer="x"), dict(schedule="x"),
                dict(lr=-1), dict(warmup_steps=-1), dict(min_lr_ratio=2.0), dict(grad_clip=-1)):
        with pytest.raises(ValueError):
            TrainConfig(**bad)


# ----- clipping ---------------------------------------------------------------- #
def test_clip_grad_norm_scales_to_max():
    a, b = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(2))
    a.grad, b.grad = torch.tensor([3.0, 0.0]), torch.tensor([0.0, 4.0])  # global norm 5
    assert clip_grad_norm([a, b], 1.0) == pytest.approx(5.0)
    new_norm = math.sqrt(float((a.grad**2).sum() + (b.grad**2).sum()))
    assert new_norm == pytest.approx(1.0, rel=1e-4)


def test_clip_grad_norm_noop_when_small_or_disabled():
    a = nn.Parameter(torch.zeros(2))
    a.grad = torch.tensor([0.3, 0.4])
    assert clip_grad_norm([a], 1.0) == pytest.approx(0.5)
    assert torch.allclose(a.grad, torch.tensor([0.3, 0.4]))
    a.grad = torch.tensor([30.0, 40.0])
    clip_grad_norm([a], 0.0)  # 0 only measures
    assert torch.allclose(a.grad, torch.tensor([30.0, 40.0]))


def test_clip_grad_norm_without_grads():
    assert clip_grad_norm([nn.Parameter(torch.zeros(1))], 1.0) == 0.0


# ----- training loop ----------------------------------------------------------- #
def test_train_yields_expected_metrics():
    model, tr, va = tiny_setup()
    cfg = TrainConfig(steps=6, batch_size=4, block_size=8, eval_interval=3, warmup_steps=2)
    metrics = list(train(model, tr, va, cfg, CPU))
    assert [m["step"] for m in metrics] == [1, 2, 3, 4, 5, 6]
    assert metrics[-1]["done"] and not metrics[0]["done"]
    assert metrics[0]["val_loss"] is not None  # evaluated at the first step
    assert metrics[1]["val_loss"] is None and metrics[2]["val_loss"] is not None
    assert metrics[-1]["val_loss"] is not None
    for m in metrics:
        assert math.isfinite(m["train_loss"]) and m["grad_norm"] >= 0 and m["tokens_per_sec"] > 0
    assert metrics[0]["lr"] < metrics[1]["lr"]  # warm-up


def test_train_without_validation_data():
    model, tr, _ = tiny_setup()
    metrics = list(train(model, tr, None, TrainConfig(steps=3, batch_size=2, block_size=8), CPU))
    assert all(m["val_loss"] is None for m in metrics)


def test_training_reduces_loss_on_real_corpus():
    text = build_corpus(True, False)
    tok = CharTokenizer.train(text)
    tr, va = train_val_split(encode_corpus(text, tok), 0.1)
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, d_model=32, n_heads=2, n_layers=1, max_seq_len=32))
    cfg = TrainConfig(steps=60, batch_size=16, block_size=32, lr=5e-3, warmup_steps=5, eval_interval=30)
    metrics = list(train(model, tr, va, cfg, CPU))
    start = math.log(tok.vocab_size)
    assert metrics[0]["train_loss"] == pytest.approx(start, abs=0.5)
    assert metrics[-1]["val_loss"] < metrics[0]["val_loss"] - 0.5


@pytest.mark.parametrize("opt", ["adam", "adamw", "sgd", "torch_adamw", "torch_sgd"])
def test_every_optimizer_runs(opt):
    model, tr, va = tiny_setup()
    cfg = TrainConfig(steps=3, batch_size=4, block_size=8, optimizer=opt, lr=1e-2)
    assert len(list(train(model, tr, va, cfg, CPU))) == 3


def test_same_seed_is_reproducible():
    def run():
        model, tr, va = tiny_setup()
        return [m["train_loss"] for m in train(model, tr, va, TrainConfig(steps=5, batch_size=4, block_size=8), CPU)]

    assert run() == run()


def test_should_stop_cancels_run():
    model, tr, va = tiny_setup()
    seen = []

    def stop():
        return len(seen) >= 2

    for m in train(model, tr, va, TrainConfig(steps=10, batch_size=2, block_size=8), CPU, should_stop=stop):
        seen.append(m)
    assert len(seen) == 2


def test_block_size_larger_than_context_rejected():
    model, tr, va = tiny_setup()
    with pytest.raises(ValueError):
        list(train(model, tr, va, TrainConfig(steps=1, block_size=64), CPU))


def test_divergence_raises():
    model, tr, va = tiny_setup()
    cfg = TrainConfig(steps=30, batch_size=4, block_size=8, optimizer="sgd", lr=1e12, grad_clip=0)
    with pytest.raises(FloatingPointError):
        list(train(model, tr, va, cfg, CPU))


def test_resume_continues_step_count():
    model, tr, va = tiny_setup()
    cfg = TrainConfig(steps=2, batch_size=2, block_size=8)
    from training.optimizers import build_optimizer

    opt = build_optimizer("adamw", model.parameters(), 1e-3)
    list(train(model, tr, va, cfg, CPU, optimizer=opt))
    resumed = list(train(model, tr, va, cfg, CPU, optimizer=opt, start_step=2))
    assert [m["step"] for m in resumed] == [3, 4]


def test_estimate_loss_is_deterministic_and_restores_mode():
    model, _, va = tiny_setup()
    model.train()
    a = estimate_loss(model, va, 2, 8, 3, CPU)
    b = estimate_loss(model, va, 2, 8, 3, CPU)
    assert a == b and model.training
