"""Pure helpers that turn the model and tokenizer into the data the pages display.

Nothing here imports Streamlit, so every function is unit tested directly. The
pages stay thin: they read widget values, call a helper, and draw the result.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from llm.attention import MultiHeadAttention
from llm.embeddings import TokenEmbedding
from llm.layers import FeedForward, LayerNorm
from llm.positional import build_positional
from llm.tokenizer import build_tokenizer
from llm.transformer import GPT, next_token_distribution

DEFAULT_TEXT = "The little robot read the book."


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
def tokenize_text(tokenizer, text: str) -> tuple[list[str], list[int]]:
    """Token strings and ids for ``text``."""
    return tokenizer.tokens(text), tokenizer.encode(text)


def replay_merges(word: str, merges: Sequence[tuple[str, str]], upto: int) -> list[str]:
    """Split ``word`` into characters, then apply the first ``upto`` merges in order.

    This replays BPE training on one word so a learner can see the pieces grow.
    """
    if upto < 0:
        raise ValueError("upto must be >= 0")
    symbols = list(word)
    for a, b in merges[:upto]:
        merged: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols


def compression_counts(corpus: str, merge_counts: Sequence[int]) -> dict[int, int]:
    """Number of tokens the corpus needs for each number of BPE merges."""
    return {m: len(build_tokenizer("bpe", corpus, m).encode(corpus)) for m in sorted(set(merge_counts))}


# --------------------------------------------------------------------------- #
# Embeddings and positions
# --------------------------------------------------------------------------- #
def clip_ids(ids: Sequence[int], max_len: int) -> list[int]:
    """Keep at most ``max_len`` ids (the model's context window)."""
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    return list(ids)[:max_len]


def embedding_demo(
    tokenizer,
    text: str,
    d_model: int,
    init: str = "normal",
    scale: bool = False,
    seed: int = 0,
    max_len: int = 32,
) -> dict:
    """Embed ``text`` with a fresh table: tokens, ids, the table and the looked-up vectors."""
    tokens = tokenizer.tokens(text)[:max_len]
    ids = clip_ids(tokenizer.encode(text), max_len)
    torch.manual_seed(seed)
    emb = TokenEmbedding(tokenizer.vocab_size, d_model, init=init, scale=scale)
    with torch.no_grad():
        vectors = emb(torch.tensor(ids, dtype=torch.long)) if ids else torch.zeros(0, d_model)
    return {"tokens": tokens, "ids": ids, "table": emb.weight.detach(), "vectors": vectors}


def positional_demo(kind: str, d_model: int, seq_len: int, seed: int = 0) -> torch.Tensor:
    """The ``(seq_len, d_model)`` positional matrix of the chosen kind."""
    torch.manual_seed(seed)
    return build_positional(kind, d_model, seq_len).encoding(seq_len).detach()


# --------------------------------------------------------------------------- #
# Attention, layer norm, feed-forward on a stand-alone (untrained) layer
# --------------------------------------------------------------------------- #
def embed_with_position(
    tokenizer, text: str, d_model: int, positional: str, seed: int, max_len: int
) -> tuple[list[str], torch.Tensor]:
    """Token strings and ``(1, T, d_model)`` input vectors (embedding + position)."""
    demo = embedding_demo(tokenizer, text, d_model, seed=seed, max_len=max_len)
    if not demo["ids"]:
        raise ValueError("the text produced no tokens")
    pos = positional_demo(positional, d_model, len(demo["ids"]), seed)
    return demo["tokens"], (demo["vectors"] + pos).unsqueeze(0)


def attention_demo(
    tokenizer,
    text: str,
    d_model: int,
    n_heads: int,
    causal: bool = True,
    positional: str = "sinusoidal",
    seed: int = 0,
    max_len: int = 24,
) -> dict:
    """Run one randomly initialised attention layer and return its internals.

    The weights are untrained, so the patterns show the *mechanics* (masking,
    row sums, how heads differ) rather than learned language structure.
    """
    tokens, x = embed_with_position(tokenizer, text, d_model, positional, seed, max_len)
    torch.manual_seed(seed + 1)
    layer = MultiHeadAttention(d_model, n_heads, causal=causal)
    layer.eval()
    with torch.no_grad():
        out, details = layer(x, return_details=True)
    return {"tokens": tokens, "input": x[0], "output": out[0], "details": {k: v[0] for k, v in details.items()}}


def layernorm_demo(
    tokenizer, text: str, d_model: int, seed: int = 0, max_len: int = 16, offset: float = 3.0, spread: float = 2.0
) -> dict:
    """Run LayerNorm on deliberately off-centre activations so the effect is obvious."""
    tokens, x = embed_with_position(tokenizer, text, d_model, "sinusoidal", seed, max_len)
    shifted = x[0] * spread + offset
    norm = LayerNorm(d_model)
    with torch.no_grad():
        stages = norm.stages(shifted)
    return {"tokens": tokens, "before": shifted, "stages": {k: v for k, v in stages.items()}}


def ffn_demo(
    tokenizer, text: str, d_model: int, ffn_mult: int, activation: str, seed: int = 0, max_len: int = 16
) -> dict:
    """Run one feed-forward layer; returns the input, hidden activations and output."""
    tokens, x = embed_with_position(tokenizer, text, d_model, "sinusoidal", seed, max_len)
    torch.manual_seed(seed + 2)
    ffn = FeedForward(d_model, ffn_mult * d_model, activation=activation)
    with torch.no_grad():
        out, hidden = ffn(x, return_hidden=True)
    return {"tokens": tokens, "input": x[0], "hidden": hidden[0], "output": out[0]}


# --------------------------------------------------------------------------- #
# The full model
# --------------------------------------------------------------------------- #
def model_forward(model: GPT, tokenizer, text: str) -> dict:
    """Forward pass over ``text`` (cropped to the context) with every intermediate tensor."""
    ids = clip_ids(tokenizer.encode(text), model.cfg.max_seq_len)
    if not ids:
        raise ValueError("the text produced no tokens")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits, details = model(torch.tensor([ids], device=device), return_details=True)
    finally:
        model.train(was_training)
    return {"tokens": tokenizer.tokens(text)[: len(ids)], "ids": ids, "logits": logits[0].cpu(), "details": details}


def top_next_tokens(
    logits: torch.Tensor,
    tokenizer,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    n: int = 10,
) -> list[dict]:
    """The ``n`` most likely next tokens from a 1-D logits vector, after sampling filters."""
    probs = next_token_distribution(logits, temperature, top_k, top_p)
    values, indices = torch.topk(probs, min(n, probs.numel()))
    return [
        {"token": tokenizer.itos[i], "id": int(i), "probability": float(v)}
        for v, i in zip(values.tolist(), indices.tolist())
        if v > 0
    ]


def generate_text(
    model: GPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> dict:
    """Continue ``prompt``; returns the full text, the new text and the new token ids.

    An empty prompt starts from token id 1 (the first real token), because the
    model needs at least one token of context.
    """
    ids = tokenizer.encode(prompt) or [1 if tokenizer.vocab_size > 1 else 0]
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    out = model.generate(
        torch.tensor([ids], device=device), max_new_tokens, temperature, top_k, top_p, generator
    )[0].tolist()
    new_ids = out[len(ids):]
    return {
        "text": tokenizer.decode(out),
        "new_text": tokenizer.decode(new_ids),
        "prompt_ids": ids,
        "new_ids": new_ids,
    }


def step_probabilities(
    model: GPT, tokenizer, ids: Sequence[int], temperature: float = 1.0,
    top_k: Optional[int] = None, top_p: Optional[float] = None,
) -> list[dict]:
    """For each position in ``ids``: probability the model gave the token that actually came next."""
    if len(ids) < 2:
        return []
    window = list(ids)[-model.cfg.max_seq_len:]
    device = next(model.parameters()).device
    with torch.no_grad():
        was_training = model.training
        model.eval()
        try:
            logits = model(torch.tensor([window], device=device))[0].cpu()
        finally:
            model.train(was_training)
    rows = []
    for pos in range(len(window) - 1):
        probs = next_token_distribution(logits[pos], temperature, top_k, top_p)
        nxt = window[pos + 1]
        rows.append(
            {
                "position": pos + 1,
                "token": tokenizer.itos[nxt],
                "probability": float(probs[nxt]),
                "rank": int((probs > probs[nxt]).sum()) + 1,
            }
        )
    return rows


def parameter_summary(model: GPT) -> dict:
    """Parameter counts (total and per component) plus an approximate memory size in MB."""
    breakdown = model.parameter_breakdown()
    return {"breakdown": breakdown, "total": breakdown["total"], "megabytes": breakdown["total"] * 4 / 1e6}
