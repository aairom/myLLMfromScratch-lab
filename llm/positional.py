"""Positional encodings: telling the model *where* each token is.

Self-attention is permutation invariant: without extra information, "dog bites
man" and "man bites dog" look identical. A positional signal of shape
``(max_len, d_model)`` is added to the token embeddings to break the symmetry.

Two variants:

* :class:`SinusoidalPositionalEncoding` - fixed, no parameters (Vaswani et al., 2017)

      PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

  Each pair of dimensions is a sine/cosine wave with its own wavelength, from
  2*pi (fast, fine position) to 10000*2*pi (slow, coarse position). Because
  sin(a+b) is a linear function of sin(a), cos(a), a shift by a fixed offset is a
  linear map, which makes relative positions easy to learn.

* :class:`LearnedPositionalEmbedding` - a trainable table, as used by GPT-2.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .embeddings import init_embedding_

POSITIONAL_KINDS = ("sinusoidal", "learned")


def sinusoidal_table(max_len: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    """Return the ``(max_len, d_model)`` sinusoidal matrix.

    ``d_model`` may be odd: the last column is then a sine column.
    """
    if max_len <= 0 or d_model <= 0:
        raise ValueError("max_len and d_model must be positive")
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
    # Even dimension indices 0, 2, 4, ... -> frequencies 1 / base^(2i/d)
    even = torch.arange(0, d_model, 2, dtype=torch.float32)  # (ceil(d/2),)
    div_term = torch.exp(-math.log(base) * even / d_model)  # (ceil(d/2),)
    angles = position * div_term  # (L, ceil(d/2))
    table = torch.zeros(max_len, d_model)
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles[:, : d_model // 2])
    return table


class SinusoidalPositionalEncoding(nn.Module):
    """Adds a fixed sinusoidal table to ``x`` of shape ``(B, T, d_model)``."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # A buffer moves with .to(device) but is not trained. It is not stored in
        # checkpoints (persistent=False) because it is recomputed from d_model/max_len.
        self.register_buffer("table", sinusoidal_table(max_len, d_model), persistent=False)

    def encoding(self, seq_len: int) -> torch.Tensor:
        """The ``(seq_len, d_model)`` positional matrix (useful for plotting)."""
        if seq_len > self.max_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_len {self.max_len}")
        return self.table[:seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding(x.size(1))


class LearnedPositionalEmbedding(nn.Module):
    """Trainable ``(max_len, d_model)`` table; position ``p`` selects row ``p``."""

    def __init__(self, d_model: int, max_len: int = 512, init: str = "normal", std: float = 0.02) -> None:
        super().__init__()
        if d_model <= 0 or max_len <= 0:
            raise ValueError("max_len and d_model must be positive")
        self.d_model = d_model
        self.max_len = max_len
        self.weight = nn.Parameter(torch.empty(max_len, d_model))
        init_embedding_(self.weight, init, std)

    def encoding(self, seq_len: int) -> torch.Tensor:
        if seq_len > self.max_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_len {self.max_len}")
        return self.weight[:seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding(x.size(1))


def build_positional(kind: str, d_model: int, max_len: int) -> nn.Module:
    """Factory used by the model config (``"sinusoidal"`` or ``"learned"``)."""
    if kind == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_len)
    if kind == "learned":
        return LearnedPositionalEmbedding(d_model, max_len)
    raise ValueError(f"unknown positional encoding {kind!r}; choose from {POSITIONAL_KINDS}")
