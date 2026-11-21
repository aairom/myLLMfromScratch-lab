"""Session state and model settings shared by every page.

This module deliberately does not import Streamlit: ``state`` is any
``MutableMapping`` (``st.session_state`` in the app, a plain ``dict`` in tests),
so the logic can be unit tested without a browser.

Lifecycle
    1. ``init_state`` fills in defaults once per browser session.
    2. The sidebar form produces a :class:`ModelSettings`; ``apply_settings``
       validates it, builds the tokenizer and a freshly initialised GPT and
       stores them in the state (training history is reset).
    3. The Training Lab trains ``state["model"]`` in place and appends to
       ``state["history"]``; ``adopt_checkpoint`` swaps in a saved model.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import MutableMapping, Optional

import torch

import config
from llm.embeddings import INIT_STRATEGIES
from llm.layers import ACTIVATIONS
from llm.positional import POSITIONAL_KINDS
from llm.tokenizer import build_tokenizer
from llm.transformer import GPT, GPTConfig
from training.data import build_corpus

TOKENIZER_KINDS = ("char", "bpe")
MAX_BPE_MERGES = 2000  # keeps retraining the tokenizer interactive


@dataclass(frozen=True)
class ModelSettings:
    """Everything the sidebar controls: data source, tokenizer and architecture."""

    tokenizer: str = "char"
    bpe_merges: int = 100
    use_bundled: bool = True
    use_input: bool = True
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 64
    ffn_mult: int = 4
    activation: str = "gelu"
    positional: str = "sinusoidal"
    embedding_init: str = "normal"
    dropout: float = 0.0
    tie_weights: bool = True
    seed: int = 42

    def problems(self) -> list[str]:
        """Human-readable validation errors (an empty list means the settings are valid)."""
        out: list[str] = []
        if self.tokenizer not in TOKENIZER_KINDS:
            out.append(f"Tokenizer must be one of {TOKENIZER_KINDS}.")
        if not 0 <= self.bpe_merges <= MAX_BPE_MERGES:
            out.append(f"BPE merges must be between 0 and {MAX_BPE_MERGES}.")
        for name in ("d_model", "n_heads", "n_layers", "max_seq_len", "ffn_mult"):
            if getattr(self, name) <= 0:
                out.append(f"{name} must be positive.")
        if self.d_model > 0 and self.n_heads > 0 and self.d_model % self.n_heads != 0:
            out.append(
                f"d_model ({self.d_model}) must be divisible by the number of heads ({self.n_heads}): "
                "every head gets d_model / heads dimensions."
            )
        if self.activation not in ACTIVATIONS:
            out.append(f"Activation must be one of {tuple(ACTIVATIONS)}.")
        if self.positional not in POSITIONAL_KINDS:
            out.append(f"Positional encoding must be one of {POSITIONAL_KINDS}.")
        if self.embedding_init not in INIT_STRATEGIES:
            out.append(f"Embedding init must be one of {INIT_STRATEGIES}.")
        if not 0.0 <= self.dropout < 1.0:
            out.append("Dropout must be in [0, 1).")
        if not (self.use_bundled or self.use_input):
            out.append("Select at least one text source (bundled corpus or files from input/).")
        return out

    def gpt_config(self, vocab_size: int) -> GPTConfig:
        """Architecture config for a given vocabulary size."""
        return GPTConfig(
            vocab_size=vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            ffn_mult=self.ffn_mult,
            activation=self.activation,
            positional=self.positional,
            embedding_init=self.embedding_init,
            tie_weights=self.tie_weights,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def default_settings() -> ModelSettings:
    """Defaults, with the seed taken from the environment configuration."""
    return ModelSettings(seed=config.load_settings().default_seed)


# Mutable defaults live in a factory so every session gets its own copies.
_STATE_DEFAULTS = {
    "model": None,  # the GPT currently shown on every page
    "tokenizer": None,
    "history": [],  # list of metric dicts from the trainer
    "train_cfg": None,  # TrainConfig of the last run
    "optimizer": None,  # kept so a run can be resumed
    "model_source": "none",  # "fresh" | "trained" | "checkpoint"
    "stop_requested": False,  # cooperative stop flag polled by the trainer
    "training": False,
    "last_error": None,
}


def init_state(state: MutableMapping) -> None:
    """Create missing keys (idempotent: existing values are never overwritten)."""
    if "settings" not in state:
        state["settings"] = default_settings()
    for key, value in _STATE_DEFAULTS.items():
        if key not in state:
            state[key] = copy.deepcopy(value)


# --------------------------------------------------------------------------- #
# Building the tokenizer and model
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def _cached_tokenizer(kind: str, text: str, merges: int):
    """Train a tokenizer once per (kind, text, merges); reruns reuse the result."""
    return build_tokenizer(kind, text, merges)


def load_corpus(settings: ModelSettings) -> str:
    """The text selected in the sidebar (raises ``ValueError`` if there is none)."""
    return build_corpus(settings.use_bundled, settings.use_input)


def make_tokenizer(settings: ModelSettings):
    """Tokenizer trained on the selected corpus; returns ``(tokenizer, corpus_text)``."""
    corpus = load_corpus(settings)
    return _cached_tokenizer(settings.tokenizer, corpus, settings.bpe_merges), corpus


def build_model(settings: ModelSettings):
    """Validate ``settings`` and return a freshly initialised ``(model, tokenizer)``.

    Raises ``ValueError`` with a readable message for any invalid combination.
    """
    problems = settings.problems()
    if problems:
        raise ValueError(" ".join(problems))
    tokenizer, _ = make_tokenizer(settings)
    cfg = settings.gpt_config(tokenizer.vocab_size)
    torch.manual_seed(settings.seed)  # same seed -> same initial weights
    return GPT(cfg), tokenizer


def apply_settings(state: MutableMapping, settings: Optional[ModelSettings] = None) -> Optional[str]:
    """Rebuild tokenizer and model from ``settings`` and reset training progress.

    Returns ``None`` on success or an error message (the old model is kept).
    """
    settings = settings if settings is not None else state["settings"]
    try:
        model, tokenizer = build_model(settings)
    except ValueError as exc:
        state["last_error"] = str(exc)
        return str(exc)
    state.update(
        settings=settings,
        model=model,
        tokenizer=tokenizer,
        history=[],
        train_cfg=None,
        optimizer=None,
        model_source="fresh",
        stop_requested=False,
        last_error=None,
    )
    return None


def ensure_model(state: MutableMapping) -> Optional[str]:
    """Build the default model on first use; returns an error message or ``None``."""
    if state.get("model") is not None and state.get("tokenizer") is not None:
        return None
    return apply_settings(state)


def settings_from_model(model: GPT, tokenizer, base: ModelSettings) -> ModelSettings:
    """Settings that describe a loaded model (so the sidebar matches it)."""
    cfg = model.cfg
    return replace(
        base,
        tokenizer=tokenizer.kind,
        bpe_merges=len(getattr(tokenizer, "merges", [])),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_seq_len=cfg.max_seq_len,
        ffn_mult=cfg.ffn_mult,
        activation=cfg.activation,
        positional=cfg.positional,
        embedding_init=cfg.embedding_init,
        dropout=cfg.dropout,
        tie_weights=cfg.tie_weights,
    )


def adopt_checkpoint(state: MutableMapping, ckpt: dict) -> None:
    """Make a loaded checkpoint (see ``training.checkpoint.load_checkpoint``) the current model."""
    model, tokenizer = ckpt["model"], ckpt["tokenizer"]
    state.update(
        settings=settings_from_model(model, tokenizer, state["settings"]),
        model=model,
        tokenizer=tokenizer,
        history=list(ckpt.get("history") or []),
        train_cfg=ckpt.get("train_config"),
        optimizer=None,
        model_source="checkpoint",
        stop_requested=False,
        last_error=None,
    )


def get_device(state: MutableMapping) -> torch.device:
    """Device chosen by the ``DEVICE`` environment setting (``auto`` picks the fastest)."""
    return config.resolve_device(config.load_settings().device)
