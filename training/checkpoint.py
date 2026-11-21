"""Saving and loading complete training state.

A checkpoint is a single ``.pt`` file holding everything needed to reproduce or
continue a run: model weights, the architecture config, the tokenizer
vocabulary (ids are meaningless without it), the optimizer state, the training
config and the metric history. Files are named ``YYYYmmdd_HHMMSS_<name>.pt`` and
never overwrite earlier ones.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

import config
from llm.tokenizer import tokenizer_from_dict
from llm.transformer import GPT, GPTConfig

from .trainer import TrainConfig

CHECKPOINT_VERSION = 1
REQUIRED_KEYS = ("version", "model_config", "model_state", "tokenizer")


class CheckpointError(Exception):
    """Raised when a checkpoint file is missing, corrupted or incompatible."""


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "model"


def new_checkpoint_path(name: str = "model", output_dir: Optional[Path] = None) -> Path:
    """Return a unique timestamped path ``<output_dir>/YYYYmmdd_HHMMSS_<name>.pt``."""
    out = Path(output_dir) if output_dir is not None else config.load_settings().output_dir
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{stamp}_{_safe_name(name)}"
    path = out / f"{base}.pt"
    counter = 1
    while path.exists():  # two saves within one second must not overwrite each other
        path = out / f"{base}_{counter}.pt"
        counter += 1
    return path


def list_checkpoints(output_dir: Optional[Path] = None) -> list[Path]:
    """Checkpoint files in ``output_dir``, newest first."""
    out = Path(output_dir) if output_dir is not None else config.load_settings().output_dir
    if not out.is_dir():
        return []
    return sorted(out.glob("*.pt"), key=lambda p: p.name, reverse=True)


def save_checkpoint(
    model: GPT,
    tokenizer,
    train_config: Optional[TrainConfig] = None,
    history: Optional[list[dict]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    name: str = "model",
    output_dir: Optional[Path] = None,
    path: Optional[Path] = None,
) -> Path:
    """Write a checkpoint and return its path (timestamped unless ``path`` is given)."""
    target = Path(path) if path is not None else new_checkpoint_path(name, output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": CHECKPOINT_VERSION,
        "model_config": model.cfg.to_dict(),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "tokenizer": tokenizer.to_dict(),
        "train_config": train_config.to_dict() if train_config else None,
        "history": list(history or []),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(state, target)
    return target


def load_checkpoint(path: Path, device: Optional[torch.device] = None) -> dict:
    """Load a checkpoint and rebuild its objects.

    Returns a dict with ``model`` (eval mode, on ``device``), ``tokenizer``,
    ``train_config`` (or None), ``history`` and ``optimizer_state`` (or None).
    Raises :class:`CheckpointError` for any problem with the file.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        # weights_only=True refuses arbitrary pickled code: safe for untrusted files.
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # corrupted or not a torch file
        raise CheckpointError(f"cannot read checkpoint {path.name}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CheckpointError("checkpoint has an unexpected structure")
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise CheckpointError(f"checkpoint is missing keys: {missing}")
    if raw["version"] != CHECKPOINT_VERSION:
        raise CheckpointError(f"unsupported checkpoint version {raw['version']}")
    try:
        model = GPT(GPTConfig.from_dict(raw["model_config"]))
        model.load_state_dict(raw["model_state"])
        tokenizer = tokenizer_from_dict(raw["tokenizer"])
        train_cfg = TrainConfig.from_dict(raw["train_config"]) if raw.get("train_config") else None
    except Exception as exc:
        raise CheckpointError(f"checkpoint contents are inconsistent: {exc}") from exc
    if model.cfg.vocab_size != tokenizer.vocab_size:
        raise CheckpointError("model vocabulary size does not match the tokenizer")
    if device is not None:
        model.to(device)
    model.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "train_config": train_cfg,
        "history": raw.get("history", []),
        "optimizer_state": raw.get("optimizer_state"),
    }
