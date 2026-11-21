"""The training loop, written as a generator so any UI can stream its metrics.

One optimisation step:

    1. sample a batch (x, y) of token windows            -> training/data.py
    2. forward pass: logits and cross-entropy loss        -> llm/transformer.py
    3. backward pass: autograd fills ``p.grad``           (backpropagation)
    4. clip the global gradient norm                      (protects against spikes)
    5. set the learning rate from the schedule
    6. optimizer.step(): update the parameters            -> training/optimizers.py

Every ``eval_interval`` steps the model is also scored on held-out validation
data. A growing gap between train and validation loss means overfitting.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Iterator, Optional

import torch
from torch import nn

from .data import get_batch
from .optimizers import OPTIMIZER_NAMES, build_optimizer
from .schedulers import SCHEDULE_NAMES, build_schedule


@dataclass
class TrainConfig:
    """Hyper-parameters of a training run (validated on creation)."""

    steps: int = 300
    batch_size: int = 16
    block_size: int = 64  # tokens per training window (must be <= model context)
    optimizer: str = "adamw"
    lr: float = 3e-3
    weight_decay: float = 0.01
    momentum: float = 0.9  # SGD only
    schedule: str = "warmup_cosine"
    warmup_steps: int = 20
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0  # 0 disables clipping
    eval_interval: int = 25
    eval_batches: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "block_size", "eval_interval", "eval_batches"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.optimizer not in OPTIMIZER_NAMES:
            raise ValueError(f"optimizer must be one of {OPTIMIZER_NAMES}")
        if self.schedule not in SCHEDULE_NAMES:
            raise ValueError(f"schedule must be one of {SCHEDULE_NAMES}")
        if self.lr < 0 or self.weight_decay < 0 or self.grad_clip < 0 or self.momentum < 0:
            raise ValueError("lr, weight_decay, momentum and grad_clip must be >= 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, state: dict) -> "TrainConfig":
        return cls(**state)


def clip_grad_norm(params: Iterable[nn.Parameter], max_norm: float) -> float:
    """Rescale gradients so their global L2 norm is at most ``max_norm``.

        total = sqrt(sum_i ||g_i||^2);  if total > max_norm: g_i *= max_norm / total

    Returns the norm *before* clipping (useful to monitor). ``max_norm <= 0``
    only measures.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total = math.sqrt(sum(float((g.detach().float() ** 2).sum()) for g in grads))
    if max_norm > 0 and total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g.mul_(scale)
    return total


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    n_batches: int,
    device: torch.device,
    seed: int = 0,
) -> float:
    """Average loss over ``n_batches`` random batches (dropout off, fixed seed)."""
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)  # same windows every call -> comparable numbers
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, block_size, gen, device)
        _, loss = model(x, y)
        total += loss.item()
    model.train(was_training)
    return total / n_batches


def train(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: Optional[torch.Tensor],
    cfg: TrainConfig,
    device: torch.device,
    should_stop: Optional[Callable[[], bool]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    start_step: int = 0,
) -> Iterator[dict]:
    """Run the training loop, yielding one metrics dict per step.

    Each dict has: ``step`` (1-based, counted from ``start_step``), ``train_loss``,
    ``val_loss`` (``None`` between evaluations), ``lr``, ``grad_norm``,
    ``tokens_per_sec``, ``elapsed`` and ``done`` (True on the last yield).
    ``should_stop`` is polled before each step so a UI can cancel a run; pass
    ``optimizer``/``start_step`` to resume from a checkpoint.
    """
    if cfg.block_size > model.cfg.max_seq_len:
        raise ValueError(
            f"block_size {cfg.block_size} exceeds the model context {model.cfg.max_seq_len}"
        )
    model.to(device)
    model.train()
    gen = torch.Generator().manual_seed(cfg.seed + start_step)
    if optimizer is None:
        optimizer = build_optimizer(
            cfg.optimizer, model.parameters(), cfg.lr, cfg.weight_decay, cfg.momentum
        )
    schedule = build_schedule(cfg.schedule, cfg.lr, cfg.warmup_steps, cfg.steps, cfg.min_lr_ratio)

    started = time.perf_counter()
    for i in range(cfg.steps):
        if should_stop is not None and should_stop():
            return
        step = start_step + i
        lr = schedule(i)
        for group in optimizer.param_groups:
            group["lr"] = lr

        t0 = time.perf_counter()
        x, y = get_batch(train_data, cfg.batch_size, cfg.block_size, gen, device)
        _, loss = model(x, y)  # forward pass + cross-entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # backpropagation
        grad_norm = clip_grad_norm(model.parameters(), cfg.grad_clip)
        optimizer.step()  # parameter update

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(
                f"loss became {loss_value} at step {step + 1}; lower the learning rate"
            )
        val_loss = None
        is_last = i == cfg.steps - 1
        if val_data is not None and ((i + 1) % cfg.eval_interval == 0 or is_last or i == 0):
            val_loss = estimate_loss(
                model, val_data, cfg.batch_size, cfg.block_size, cfg.eval_batches, device
            )
        dt = max(time.perf_counter() - t0, 1e-9)
        yield {
            "step": step + 1,
            "train_loss": loss_value,
            "val_loss": val_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "tokens_per_sec": cfg.batch_size * cfg.block_size / dt,
            "elapsed": time.perf_counter() - started,
            "done": is_last,
        }
