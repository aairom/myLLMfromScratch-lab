"""A decoder-only transformer (GPT) assembled from the hand-written parts.

Data flow for one forward pass (B = batch, T = sequence, D = d_model, V = vocab):

    ids (B, T)
      -> token embedding        (B, T, D)
      -> + positional encoding  (B, T, D)
      -> N x TransformerBlock   (B, T, D)    # the "residual stream"
      -> final LayerNorm        (B, T, D)
      -> linear head            (B, T, V)    # one score ("logit") per vocab token
      -> softmax / cross-entropy

Each :class:`TransformerBlock` uses the *pre-norm* layout:

    x = x + Attention(LayerNorm(x))     # tokens exchange information
    x = x + FeedForward(LayerNorm(x))   # each token is processed on its own

The additions are *residual connections*: every block only writes a small
correction into the stream, which keeps gradients flowing through deep stacks.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import nn

from .attention import MultiHeadAttention
from .embeddings import INIT_STRATEGIES, TokenEmbedding
from .layers import ACTIVATIONS, FeedForward, LayerNorm
from .positional import POSITIONAL_KINDS, build_positional


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class GPTConfig:
    """All architecture hyper-parameters (validated on creation)."""

    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 64  # context window ("block size")
    dropout: float = 0.0
    ffn_mult: int = 4  # FFN hidden width = ffn_mult * d_model
    activation: str = "gelu"
    positional: str = "sinusoidal"
    embedding_init: str = "normal"
    tie_weights: bool = True  # share the token embedding with the output head

    def __post_init__(self) -> None:
        for name in ("vocab_size", "d_model", "n_heads", "n_layers", "max_seq_len", "ffn_mult"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {tuple(ACTIVATIONS)}")
        if self.positional not in POSITIONAL_KINDS:
            raise ValueError(f"positional must be one of {POSITIONAL_KINDS}")
        if self.embedding_init not in INIT_STRATEGIES:
            raise ValueError(f"embedding_init must be one of {INIT_STRATEGIES}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, state: dict) -> "GPTConfig":
        return cls(**state)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of the correct tokens, written by hand.

        loss = -1/N * sum_i log softmax(logits_i)[target_i]

    ``log softmax(z) = z - logsumexp(z)`` is computed in a numerically stable way.
    Shapes: ``logits (N, V)``, ``targets (N,)`` of ids. A model that guesses
    uniformly scores ``ln(V)``; lower is better.
    """
    if logits.dim() != 2 or targets.shape != (logits.size(0),):
        raise ValueError("expected logits (N, V) and targets (N,)")
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return nll.mean()


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """One pre-norm decoder block: attention sub-layer then feed-forward sub-layer."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, causal=True)
        self.ln2 = LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.ffn_mult * cfg.d_model, cfg.dropout, cfg.activation)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        if not return_details:
            x = x + self.attn(self.ln1(x))
            return x + self.ffn(self.ln2(x))

        normed1 = self.ln1(x)
        attn_out, attn_details = self.attn(normed1, return_details=True)
        after_attn = x + attn_out
        normed2 = self.ln2(after_attn)
        ffn_out, hidden = self.ffn(normed2, return_hidden=True)
        out = after_attn + ffn_out
        details = {
            "input": x.detach(),
            "ln1": normed1.detach(),
            "attention": attn_details,  # q, k, v, scores, weights, head_output
            "attn_out": attn_out.detach(),
            "after_attn": after_attn.detach(),
            "ln2": normed2.detach(),
            "ffn_hidden": hidden,
            "ffn_out": ffn_out.detach(),
            "output": out.detach(),
        }
        return out, details


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #
def next_token_distribution(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Turn raw logits ``(..., V)`` into the probabilities used for sampling.

    * ``temperature`` divides the logits: <1 sharpens, >1 flattens.
      ``0`` means greedy (all mass on the arg-max).
    * ``top_k`` keeps only the ``k`` highest-scoring tokens.
    * ``top_p`` (nucleus) keeps the smallest set of tokens whose cumulative
      probability reaches ``top_p`` (always at least one token).
    """
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    if temperature == 0:  # greedy
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs

    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Drop a token if the probability mass *before* it already reaches top_p.
        remove_sorted = (cumulative - sorted_probs) >= top_p
        remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
        probs = probs.masked_fill(remove, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


# --------------------------------------------------------------------------- #
# GPT model
# --------------------------------------------------------------------------- #
class GPT(nn.Module):
    """Decoder-only language model predicting the next token at every position."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = TokenEmbedding(cfg.vocab_size, cfg.d_model, init=cfg.embedding_init)
        self.pos = build_positional(cfg.positional, cfg.d_model, cfg.max_seq_len)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_f = LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:  # output scores = similarity to the token embeddings
            self.lm_head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self) -> None:
        """GPT-2 style: N(0, 0.02) for linears; residual outputs shrunk by sqrt(2N)."""
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        res_std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=res_std)
            nn.init.normal_(block.ffn.fc_out.weight, mean=0.0, std=res_std)
        if not self.cfg.tie_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        # A tied head keeps the user-chosen embedding initialisation.

    def forward(
        self,
        ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        """Return ``logits`` (B, T, V), or ``(logits, loss)`` if ``targets`` are given.

        With ``return_details=True`` a dict of detached intermediate tensors is
        appended to the result (token embeddings, per-block internals, logits).
        """
        if ids.dim() != 2:
            raise ValueError(f"ids must have shape (B, T), got {tuple(ids.shape)}")
        t = ids.size(1)
        if t == 0:
            raise ValueError("sequence length must be > 0")
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")

        tok = self.tok_emb(ids)  # (B, T, D)
        with_pos = self.pos(tok)  # add position information
        x = self.drop(with_pos)
        block_details = []
        for block in self.blocks:
            if return_details:
                x, d = block(x, return_details=True)
                block_details.append(d)
            else:
                x = block(x)
        x_norm = self.ln_f(x)
        logits = self.lm_head(x_norm)  # (B, T, V)

        loss = None
        if targets is not None:
            if targets.shape != ids.shape:
                raise ValueError("targets must have the same shape as ids")
            loss = cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        result = logits if loss is None else (logits, loss)
        if not return_details:
            return result
        details = {
            "token_embeddings": tok.detach(),
            "with_position": with_pos.detach(),
            "blocks": block_details,
            "final_norm": x_norm.detach(),
            "logits": logits.detach(),
        }
        return (result, details) if loss is None else (*result, details)

    # ----- introspection ---------------------------------------------------- #
    def num_parameters(self) -> int:
        """Total trainable parameters (a tied matrix is counted once)."""
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> dict[str, int]:
        """Parameter counts per component; they add up to ``num_parameters()``."""
        seen: set[int] = set()

        def count(module: nn.Module) -> int:
            total = 0
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    total += p.numel()
            return total

        out = {"token_embedding": count(self.tok_emb), "positional": count(self.pos)}
        out["attention"] = sum(count(b.attn) for b in self.blocks)
        out["feed_forward"] = sum(count(b.ffn) for b in self.blocks)
        out["layer_norm"] = sum(count(b.ln1) + count(b.ln2) for b in self.blocks) + count(self.ln_f)
        out["output_head"] = count(self.lm_head)  # 0 when tied: already counted above
        out["total"] = sum(out.values())
        return out

    # ----- generation ------------------------------------------------------- #
    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Autoregressively extend ``ids`` (B, T) by ``max_new_tokens`` tokens.

        Loop: crop to the context window -> forward -> take the last position's
        logits -> adjust (temperature, top-k, top-p) -> sample -> append.
        Pass a CPU ``torch.Generator`` for reproducible sampling.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if ids.dim() != 2 or ids.size(1) == 0:
            raise ValueError("ids must have shape (B, T) with T > 0")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = ids[:, -self.cfg.max_seq_len :]
                logits = self(context)[:, -1, :]
                probs = next_token_distribution(logits, temperature, top_k, top_p)
                # Sample on CPU so a CPU generator works for every device (incl. MPS).
                nxt = torch.multinomial(probs.float().cpu(), 1, generator=generator).to(ids.device)
                ids = torch.cat([ids, nxt], dim=1)
        finally:
            self.train(was_training)
        return ids
