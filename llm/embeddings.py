"""Token embeddings: the lookup table that turns ids into vectors.

Each token id selects one row of a learnable matrix ``E`` with shape
``(vocab_size, d_model)``. Row ``i`` *is* the vector for token ``i``. Training
nudges these rows so that tokens used in similar contexts end up close together.

This is written by hand (no ``nn.Embedding``) to show that an embedding is just
indexing into a parameter matrix; autograd routes gradients only to the rows
that were looked up.
"""
from __future__ import annotations

import math

import torch
from torch import nn

INIT_STRATEGIES = ("normal", "uniform", "xavier")


def init_embedding_(weight: torch.Tensor, strategy: str = "normal", std: float = 0.02) -> torch.Tensor:
    """Initialise ``weight`` in place and return it.

    * ``normal``  - N(0, std^2). GPT-style default (std = 0.02).
    * ``uniform`` - U(-a, a) with ``a = std * sqrt(3)`` so the variance equals std^2.
    * ``xavier``  - Glorot uniform, scaled by fan-in + fan-out of the matrix.
    """
    if strategy not in INIT_STRATEGIES:
        raise ValueError(f"unknown init strategy {strategy!r}; choose from {INIT_STRATEGIES}")
    with torch.no_grad():
        if strategy == "normal":
            weight.normal_(mean=0.0, std=std)
        elif strategy == "uniform":
            bound = std * math.sqrt(3.0)
            weight.uniform_(-bound, bound)
        else:  # xavier
            fan_out, fan_in = weight.shape
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            weight.uniform_(-bound, bound)
    return weight


class TokenEmbedding(nn.Module):
    """Learnable lookup table ``(vocab_size, d_model)``.

    Args:
        vocab_size: number of distinct tokens.
        d_model: size of every embedding vector.
        init: one of :data:`INIT_STRATEGIES`.
        std: standard deviation used by ``normal`` / ``uniform`` init.
        scale: if True, multiply outputs by ``sqrt(d_model)`` (as in the
            original Transformer) so embeddings are not drowned out by
            positional encodings of magnitude ~1.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        init: str = "normal",
        std: float = 0.02,
        scale: bool = False,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("vocab_size and d_model must be positive")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        init_embedding_(self.weight, init, std)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``(...,) int ids -> (..., d_model)`` by indexing rows of the table."""
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise IndexError(f"token id out of range [0, {self.vocab_size})")
        out = self.weight[ids]  # advanced indexing = the whole "layer"
        return out * math.sqrt(self.d_model) if self.scale else out
