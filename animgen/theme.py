"""Colours, easing curves and small numeric helpers shared by every scene.

The palette is colour-blind friendly (blue / orange / green / purple / red) and
each *role* keeps one colour in every animation: queries are always orange, keys
always blue, values always green, and so on. A learner who sees orange in one GIF
can therefore expect the same meaning in the next.
"""
from __future__ import annotations

import math
from typing import Sequence

RGB = tuple[int, int, int]

BG: RGB = (250, 251, 253)
PANEL: RGB = (255, 255, 255)
INK: RGB = (31, 35, 40)
MUTED: RGB = (96, 105, 115)
GRID: RGB = (222, 226, 232)

TOKEN: RGB = (59, 130, 212)  # token ids, keys, inputs
QUERY: RGB = (230, 126, 34)  # queries, the thing being looked at
VALUE: RGB = (39, 174, 96)  # values, outputs, "good"
ACCENT: RGB = (124, 92, 216)  # position, residual stream
LOSS: RGB = (214, 69, 65)  # loss, errors, gradients
GOLD: RGB = (241, 196, 15)

ROLE_COLOURS = {
    "token": TOKEN, "key": TOKEN, "query": QUERY, "value": VALUE,
    "position": ACCENT, "loss": LOSS, "ink": INK,
}


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease(t: float) -> float:
    """Smooth-step: slow start, slow stop (``3t^2 - 2t^3``)."""
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def ease_out(t: float) -> float:
    t = clamp01(t)
    return 1.0 - (1.0 - t) ** 3


def segment(t: float, start: float, end: float) -> float:
    """Progress of ``t`` inside ``[start, end]``, clamped to 0..1 (0 before, 1 after)."""
    if end <= start:
        return 1.0 if t >= end else 0.0
    return clamp01((t - start) / (end - start))


def mix(a: RGB, b: RGB, t: float) -> RGB:
    """Linear blend of two colours (``t = 0`` gives ``a``)."""
    t = clamp01(t)
    return (round(lerp(a[0], b[0], t)), round(lerp(a[1], b[1], t)), round(lerp(a[2], b[2], t)))


def tint(colour: RGB, amount: float) -> RGB:
    """Pale version of ``colour`` (``amount`` = 1 keeps it, 0 gives white)."""
    return mix((255, 255, 255), colour, amount)


def with_alpha(colour: RGB, alpha: float) -> tuple[int, int, int, int]:
    return (*colour, round(255 * clamp01(alpha)))


def heat(value: float, vmax: float = 1.0, low: RGB = (247, 250, 255), high: RGB = TOKEN) -> RGB:
    """Colour of a non-negative value on a white-to-colour scale."""
    return mix(low, high, value / vmax if vmax > 0 else 0.0)


def diverging(value: float, vmax: float = 1.0) -> RGB:
    """Blue for negative, white for zero, orange for positive (for signed matrices)."""
    t = max(-1.0, min(1.0, value / vmax if vmax > 0 else 0.0))
    return mix((255, 255, 255), QUERY if t >= 0 else TOKEN, abs(t))


def spread(n: int, start: float, end: float) -> list[float]:
    """``n`` evenly spaced numbers from ``start`` to ``end`` (a single one sits in the middle)."""
    if n <= 0:
        return []
    if n == 1:
        return [(start + end) / 2]
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def sparkline(values: Sequence[float], size: int) -> list[float]:
    """Resample ``values`` to ``size`` points by averaging equal chunks (for compact plots)."""
    if size <= 0 or not values:
        return []
    out = []
    n = len(values)
    for i in range(size):
        lo, hi = i * n // size, max(i * n // size + 1, (i + 1) * n // size)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def round_half(x: float) -> int:
    return int(math.floor(x + 0.5))
