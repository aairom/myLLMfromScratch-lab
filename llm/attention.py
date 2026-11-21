"""Self-attention and multi-head attention, written from scratch.

Intuition
    Every token asks a *question* (query), advertises what it *contains* (key)
    and carries a *payload* (value). A token's output is a weighted average of
    all value vectors, where the weight is how well its query matches each key:

        Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    * ``/ sqrt(d_k)`` keeps dot products from growing with the dimension, which
      would push softmax into saturated regions with vanishing gradients.
    * A *causal mask* hides future positions, so token ``t`` only attends to
      tokens ``<= t``. This is what makes a decoder-only model a valid
      next-token predictor.

Multi-head attention runs ``n_heads`` attentions in parallel on lower-dimensional
slices (``d_k = d_model / n_heads``). Each head can learn a different relation
(e.g. "previous token", "same word earlier", ...); the heads are concatenated and
mixed by an output projection.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Lower-triangular boolean mask ``(T, T)``; ``True`` = position may be attended."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def attention_scores(q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Scaled, optionally masked scores ``Q K^T / sqrt(d_k)`` (before softmax).

    Shapes: ``q`` and ``k`` are ``(..., T, d_k)``; result is ``(..., T, T)``.
    Masked positions (``mask == False``) are set to ``-inf`` so softmax gives
    them exactly zero weight.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(output, weights)``.

    ``weights`` has shape ``(..., T, T)`` and every row sums to 1 (rows that are
    entirely masked become all zeros instead of NaN). ``output`` is
    ``weights @ v`` with shape ``(..., T, d_v)``.
    """
    scores = attention_scores(q, k, mask)
    weights = torch.softmax(scores, dim=-1)  # normalise each query's row
    weights = torch.nan_to_num(weights, nan=0.0)  # fully-masked rows -> zeros
    used = dropout(weights) if dropout is not None else weights
    return used @ v, weights


class MultiHeadAttention(nn.Module):
    """Multi-head (causal) self-attention.

    Args:
        d_model: model width; must be divisible by ``n_heads``.
        n_heads: number of parallel attention heads.
        dropout: dropout on attention weights and on the output projection.
        causal: if True, apply the lower-triangular mask.
        bias: use bias terms in the linear projections.

    Shapes (B = batch, T = sequence, H = heads, d_k = d_model / H):
        input  x          (B, T, d_model)
        q, k, v           (B, H, T, d_k)
        weights           (B, H, T, T)
        output            (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        causal: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if n_heads <= 0 or d_model <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        # Separate projections keep the three roles visible (a fused QKV matrix is an optimisation).
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, d_model) -> (B, H, T, d_k)``: carve the width into heads."""
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, H, T, d_k) -> (B, T, d_model)``: concatenate the heads."""
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, self.d_model)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        """Apply attention to ``x``.

        With ``return_details=True`` returns ``(output, details)`` where
        ``details`` holds detached ``q, k, v, scores, weights, head_output`` for
        visualisation (``scores`` are pre-softmax, with ``-inf`` where masked).
        """
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"expected input of shape (B, T, {self.d_model}), got {tuple(x.shape)}")
        t = x.size(1)
        if t == 0:
            raise ValueError("sequence length must be > 0")

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        mask = causal_mask(t, x.device) if self.causal else None
        head_out, weights = scaled_dot_product_attention(q, k, v, mask, self.attn_dropout)
        out = self.resid_dropout(self.out_proj(self._merge_heads(head_out)))

        if not return_details:
            return out
        details = {
            "q": q.detach(),
            "k": k.detach(),
            "v": v.detach(),
            "scores": attention_scores(q.detach(), k.detach(), mask),
            "weights": weights.detach(),
            "head_output": head_out.detach(),
        }
        return out, details
