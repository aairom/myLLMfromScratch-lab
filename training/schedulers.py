"""Learning-rate schedules as pure functions of the step number.

A constant learning rate is rarely best:

* at the very start the Adam statistics are noisy and the weights are random, so
  large steps can wreck training -> **linear warm-up** from ~0 to ``base_lr``;
* near the end smaller steps let the model settle into a minimum ->
  **cosine decay** from ``base_lr`` down to ``min_lr``:

      lr(t) = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress)),
      progress = (t - warmup) / (total - warmup)
"""
from __future__ import annotations

import math
from typing import Callable

SCHEDULE_NAMES = ("warmup_cosine", "constant")


def warmup_cosine_lr(
    step: int,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Learning rate for 0-based ``step``.

    ``step < warmup_steps``: linear ramp ``base_lr * (step + 1) / warmup_steps``.
    Afterwards: cosine decay to ``base_lr * min_lr_ratio`` at ``total_steps`` (held afterwards).
    """
    if base_lr < 0 or not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("base_lr must be >= 0 and min_lr_ratio in [0, 1]")
    if warmup_steps < 0 or total_steps <= 0:
        raise ValueError("warmup_steps must be >= 0 and total_steps > 0")
    if step < 0:
        raise ValueError("step must be >= 0")
    min_lr = base_lr * min_lr_ratio
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    decay_span = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_span
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def constant_lr(step: int, base_lr: float) -> float:
    """Always ``base_lr`` (for comparison with the warm-up + cosine schedule)."""
    if step < 0:
        raise ValueError("step must be >= 0")
    return base_lr


def build_schedule(
    name: str,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> Callable[[int], float]:
    """Return ``schedule(step) -> lr`` for the named schedule."""
    if name == "warmup_cosine":
        return lambda step: warmup_cosine_lr(step, base_lr, warmup_steps, total_steps, min_lr_ratio)
    if name == "constant":
        return lambda step: constant_lr(step, base_lr)
    raise ValueError(f"unknown schedule {name!r}; choose from {SCHEDULE_NAMES}")
