"""Training-lab logic without Streamlit: data preparation, running a session, checkpoints.

``run_training`` is a generator over the state, so the page can redraw charts
after every few steps while tests simply exhaust it.
"""
from __future__ import annotations

from typing import Iterator, MutableMapping, Optional

import torch

from training.checkpoint import CheckpointError, list_checkpoints, load_checkpoint, save_checkpoint
from training.data import encode_corpus, train_val_split
from training.optimizers import build_optimizer
from training.schedulers import build_schedule
from training.trainer import TrainConfig, train
from ui.state import adopt_checkpoint, get_device, load_corpus

MIN_VAL_TOKENS = 8


def prepare_data(state: MutableMapping, val_fraction: float = 0.1) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Encode the selected corpus and split it into train / validation tensors.

    Raises ``ValueError`` with a clear message if there is too little text for
    the model's context window.
    """
    corpus = load_corpus(state["settings"])
    data = encode_corpus(corpus, state["tokenizer"])
    block = state["model"].cfg.max_seq_len
    if len(data) < block + 2:
        raise ValueError(
            f"The text has only {len(data)} tokens, fewer than the context window ({block}) + 2. "
            "Add text to input/, use a character tokenizer, or reduce the context window."
        )
    train_data, val_data = train_val_split(data, val_fraction)
    if len(val_data) < MIN_VAL_TOKENS or len(val_data) < 2:
        return data, None  # too small to validate on: train on everything
    if len(train_data) < block + 2:
        return data, None
    return train_data, val_data


def check_train_config(state: MutableMapping, cfg: TrainConfig) -> list[str]:
    """Problems that would stop a run (empty list = fine)."""
    problems = []
    context = state["model"].cfg.max_seq_len
    if cfg.block_size > context:
        problems.append(f"Window length ({cfg.block_size}) is larger than the model context ({context}).")
    return problems


def schedule_preview(cfg: TrainConfig):
    """The ``schedule(step)`` function for the chosen settings (for the preview chart)."""
    return build_schedule(cfg.schedule, cfg.lr, cfg.warmup_steps, cfg.steps, cfg.min_lr_ratio)


def run_training(state: MutableMapping, cfg: TrainConfig, resume: bool = False) -> Iterator[dict]:
    """Train ``state['model']`` and append every metrics dict to ``state['history']``.

    Honors ``state['stop_requested']`` between steps. On ``resume`` the existing
    optimizer (and step counter) are reused. Errors from the trainer
    (``FloatingPointError``, ``ValueError``) propagate to the caller.
    """
    model, device = state["model"], get_device(state)
    train_data, val_data = prepare_data(state)
    start_step = len(state["history"]) if resume else 0
    if not resume:
        state["history"] = []
        state["optimizer"] = None
    optimizer = state.get("optimizer") if resume else None
    if optimizer is None:
        model.to(device)  # parameters must be on the device before the optimizer sees them
        optimizer = build_optimizer(cfg.optimizer, model.parameters(), cfg.lr, cfg.weight_decay, cfg.momentum)
    state.update(optimizer=optimizer, train_cfg=cfg, stop_requested=False, training=True)
    try:
        for metrics in train(
            model, train_data, val_data, cfg, device,
            should_stop=lambda: bool(state.get("stop_requested")),
            optimizer=optimizer, start_step=start_step,
        ):
            state["history"].append(metrics)
            yield metrics
    finally:
        state["training"] = False
        model.eval()
        if state["history"]:
            state["model_source"] = "trained"


def finish_message(state: MutableMapping) -> str:
    """One-sentence outcome of the last run."""
    history = state["history"]
    if not history:
        return "No steps were run."
    last = history[-1]
    first = history[0]["train_loss"]
    stopped = "Stopped" if state.get("stop_requested") else "Finished"
    return (
        f"{stopped} after {last['step']} steps: training loss {first:.3f} -> {last['train_loss']:.3f} "
        f"in {last['elapsed']:.1f}s."
    )


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_current(state: MutableMapping, name: str = "model"):
    """Save the current model, tokenizer, config, history and optimizer to ``output/``."""
    if state.get("model") is None:
        raise ValueError("There is no model to save.")
    return save_checkpoint(
        state["model"], state["tokenizer"], state.get("train_cfg"), state["history"],
        state.get("optimizer"), name=name,
    )


def available_checkpoints() -> list:
    return list_checkpoints()


def load_into_state(state: MutableMapping, path) -> Optional[str]:
    """Load a checkpoint into the session; returns an error message or ``None``."""
    try:
        ckpt = load_checkpoint(path, get_device(state))
    except CheckpointError as exc:
        state["last_error"] = str(exc)
        return str(exc)
    adopt_checkpoint(state, ckpt)
    return None


def history_summary(history: list[dict]) -> dict:
    """Headline numbers for the metrics panel."""
    if not history:
        return {}
    vals = [h["val_loss"] for h in history if h.get("val_loss") is not None]
    last = history[-1]
    return {
        "step": last["step"],
        "train_loss": last["train_loss"],
        "val_loss": vals[-1] if vals else None,
        "best_val_loss": min(vals) if vals else None,
        "lr": last["lr"],
        "tokens_per_sec": last["tokens_per_sec"],
        "elapsed": last["elapsed"],
    }
