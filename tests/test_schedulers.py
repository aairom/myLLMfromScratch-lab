"""Unit tests for learning-rate schedules."""
import math

import pytest

from training.schedulers import build_schedule, constant_lr, warmup_cosine_lr

BASE, WARM, TOTAL = 1.0, 10, 100


def lr(step, ratio=0.1):
    return warmup_cosine_lr(step, BASE, WARM, TOTAL, ratio)


def test_warmup_is_linear_and_reaches_base():
    assert lr(0) == pytest.approx(0.1)
    assert lr(4) == pytest.approx(0.5)
    assert lr(WARM - 1) == pytest.approx(BASE)


def test_cosine_starts_at_base_and_ends_at_min():
    assert lr(WARM) == pytest.approx(BASE)
    assert lr(TOTAL) == pytest.approx(0.1)
    assert lr(TOTAL + 50) == pytest.approx(0.1)  # held after the end


def test_cosine_midpoint():
    mid = WARM + (TOTAL - WARM) // 2
    assert lr(mid) == pytest.approx(0.1 + 0.5 * 0.9 * (1 + math.cos(math.pi * 0.5)))


def test_monotonic_decay_after_warmup():
    values = [lr(s) for s in range(WARM, TOTAL + 1)]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))


def test_no_warmup():
    assert warmup_cosine_lr(0, 2.0, 0, 50) == pytest.approx(2.0)


def test_warmup_longer_than_total():
    # Degenerate config: warm-up takes priority, so the rate keeps ramping
    # (step 5 -> 6/20, step 10 -> 11/20) and never reaches the decay phase.
    assert warmup_cosine_lr(5, 1.0, 20, 10) == pytest.approx(0.3)
    assert warmup_cosine_lr(10, 1.0, 20, 10) == pytest.approx(0.55)


def test_min_ratio_zero_and_one():
    assert warmup_cosine_lr(TOTAL, 1.0, WARM, TOTAL, 0.0) == pytest.approx(0.0)
    assert warmup_cosine_lr(TOTAL - 1, 1.0, WARM, TOTAL, 1.0) == pytest.approx(1.0)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        warmup_cosine_lr(-1, 1.0, 5, 10)
    with pytest.raises(ValueError):
        warmup_cosine_lr(0, 1.0, -1, 10)
    with pytest.raises(ValueError):
        warmup_cosine_lr(0, 1.0, 5, 0)
    with pytest.raises(ValueError):
        warmup_cosine_lr(0, -1.0, 5, 10)
    with pytest.raises(ValueError):
        warmup_cosine_lr(0, 1.0, 5, 10, 1.5)
    with pytest.raises(ValueError):
        constant_lr(-1, 1.0)


def test_build_schedule():
    cosine = build_schedule("warmup_cosine", 1.0, 10, 100)
    assert cosine(0) == pytest.approx(0.1)
    const = build_schedule("constant", 0.3, 10, 100)
    assert const(0) == const(99) == 0.3
    with pytest.raises(ValueError):
        build_schedule("linear", 1.0, 1, 2)
