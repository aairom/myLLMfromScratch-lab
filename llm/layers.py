"""Layer normalisation and the position-wise feed-forward network.

LayerNorm
    Normalises each token's feature vector independently so activations keep a
    stable scale as depth grows, which makes deep stacks trainable:

        mu    = mean(x)                       (over the feature axis)
        var   = mean((x - mu)^2)              (biased variance)
        x_hat = (x - mu) / sqrt(var + eps)
        y     = gamma * x_hat + beta          (learnable scale and shift)

    Unlike BatchNorm it does not depend on the batch, so it behaves identically
    during training and inference and works for any sequence length.

Feed-forward network (FFN)
    Attention *mixes information between tokens*; the FFN *transforms each token
    on its own*: expand to ``4 * d_model``, apply a non-linearity, project back.
    Most of a transformer's parameters live here.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class LayerNorm(nn.Module):
    """Layer normalisation over the last dimension, written by hand."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # gamma
        self.bias = nn.Parameter(torch.zeros(d_model))  # beta

    def stages(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return every intermediate quantity (for the educational breakdown)."""
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)  # biased, like nn.LayerNorm
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        output = normalized * self.weight + self.bias
        return {"mean": mean, "var": var, "normalized": normalized, "output": output}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.d_model:
            raise ValueError(f"expected last dimension {self.d_model}, got {x.size(-1)}")
        return self.stages(x)["output"]


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU (tanh approximation): a smooth ReLU, ``x * Phi(x)``.

        0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def relu(x: torch.Tensor) -> torch.Tensor:
    """Rectified linear unit: ``max(0, x)``."""
    return torch.clamp(x, min=0.0)


ACTIVATIONS = {"gelu": gelu, "relu": relu}


class FeedForward(nn.Module):
    """``Linear(d_model -> d_hidden) -> activation -> Linear(d_hidden -> d_model)``.

    Args:
        d_model: input/output width.
        d_hidden: inner width; defaults to ``4 * d_model``.
        dropout: applied to the output.
        activation: ``"gelu"`` or ``"relu"``.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {tuple(ACTIVATIONS)}")
        d_hidden = d_hidden if d_hidden is not None else 4 * d_model
        if d_model <= 0 or d_hidden <= 0:
            raise ValueError("d_model and d_hidden must be positive")
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.activation = ACTIVATIONS[activation]
        self.fc_in = nn.Linear(d_model, d_hidden)
        self.fc_out = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        """Apply the FFN; optionally also return the hidden activations ``(B, T, d_hidden)``."""
        hidden = self.activation(self.fc_in(x))
        out = self.dropout(self.fc_out(hidden))
        return (out, hidden.detach()) if return_hidden else out
