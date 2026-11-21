"""Figures for training metrics and for the model's data flow."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from llm.transformer import GPTConfig

from .common import empty_figure


# --------------------------------------------------------------------------- #
# Training curves
# --------------------------------------------------------------------------- #
def moving_average(values: Sequence[float], window: int) -> list[float]:
    """Trailing moving average (the first points average over fewer samples)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    arr = np.asarray(values, dtype=float)
    out = []
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out.append(float(arr[lo : i + 1].mean()))
    return out


def _series(history: Sequence[dict], key: str) -> tuple[list[int], list[float]]:
    """Steps and values of ``key``, skipping entries where it is missing/None/NaN."""
    xs, ys = [], []
    for row in history:
        v = row.get(key)
        if v is not None and math.isfinite(v):
            xs.append(row["step"])
            ys.append(v)
    return xs, ys


def loss_figure(
    history: Sequence[dict],
    smooth_window: int = 1,
    baseline: Optional[float] = None,
    log_y: bool = False,
) -> go.Figure:
    """Training and validation loss versus step.

    ``baseline`` draws the loss of a uniform guesser, ``ln(vocab_size)``.
    """
    steps, train = _series(history, "train_loss")
    if not steps:
        return empty_figure("Loss curve appears here once training starts")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=train, mode="lines", name="train (raw)", opacity=0.35 if smooth_window > 1 else 1.0))
    if smooth_window > 1:
        fig.add_trace(go.Scatter(x=steps, y=moving_average(train, smooth_window), mode="lines", name=f"train (avg {smooth_window})"))
    vx, vy = _series(history, "val_loss")
    if vx:
        fig.add_trace(go.Scatter(x=vx, y=vy, mode="lines+markers", name="validation"))
    if baseline is not None:
        fig.add_hline(y=baseline, line_dash="dot", annotation_text="uniform guess ln(V)")
    fig.update_layout(
        title="Cross-entropy loss",
        xaxis_title="step",
        yaxis_title="loss",
        yaxis_type="log" if log_y else "linear",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _line_figure(history: Sequence[dict], key: str, title: str, y_title: str, empty_msg: str) -> go.Figure:
    xs, ys = _series(history, key)
    if not xs:
        return empty_figure(empty_msg)
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", name=key))
    fig.update_layout(title=title, xaxis_title="step", yaxis_title=y_title, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def lr_figure(history: Sequence[dict]) -> go.Figure:
    """Learning rate versus step (warm-up ramp, then cosine decay)."""
    return _line_figure(history, "lr", "Learning rate", "lr", "Learning-rate curve appears once training starts")


def grad_norm_figure(history: Sequence[dict]) -> go.Figure:
    """Global gradient norm (before clipping) versus step."""
    return _line_figure(history, "grad_norm", "Gradient norm", "L2 norm", "Gradient norm appears once training starts")


def throughput_figure(history: Sequence[dict]) -> go.Figure:
    """Training throughput in tokens per second."""
    return _line_figure(history, "tokens_per_sec", "Throughput", "tokens / s", "Throughput appears once training starts")


def schedule_figure(schedule, total_steps: int) -> go.Figure:
    """Preview a learning-rate schedule ``schedule(step) -> lr`` before training."""
    if total_steps <= 0:
        return empty_figure("steps must be positive")
    steps = list(range(total_steps))
    fig = go.Figure(go.Scatter(x=steps, y=[schedule(s) for s in steps], mode="lines"))
    fig.update_layout(title="Learning-rate schedule preview", xaxis_title="step", yaxis_title="lr", margin=dict(l=40, r=20, t=50, b=40))
    return fig


# --------------------------------------------------------------------------- #
# Data flow
# --------------------------------------------------------------------------- #
def model_stages(cfg: GPTConfig, batch: int = 1, seq: Optional[int] = None) -> list[tuple[str, str, tuple]]:
    """``(name, description, tensor_shape)`` for each stage of a full forward pass."""
    t = seq if seq is not None else cfg.max_seq_len
    b, d, v = batch, cfg.d_model, cfg.vocab_size
    return [
        ("Token ids", "integers from the tokenizer", (b, t)),
        ("Token embedding", "table lookup: id -> vector", (b, t, d)),
        ("+ Positional encoding", f"{cfg.positional}: adds order information", (b, t, d)),
        (f"Transformer block x {cfg.n_layers}", "attention + feed-forward with residuals", (b, t, d)),
        ("Final LayerNorm", "normalise each token vector", (b, t, d)),
        ("Linear head -> logits", "one score per vocabulary token", (b, t, v)),
        ("Softmax -> probabilities", "next-token distribution", (b, t, v)),
    ]


def block_stages(cfg: GPTConfig, batch: int = 1, seq: Optional[int] = None) -> list[tuple[str, str, tuple]]:
    """Stages inside one pre-norm transformer block."""
    t = seq if seq is not None else cfg.max_seq_len
    b, d, h = batch, cfg.d_model, cfg.n_heads
    return [
        ("Input x", "residual stream", (b, t, d)),
        ("LayerNorm 1", "normalise before attention", (b, t, d)),
        (f"Multi-head attention ({h} heads)", f"Q, K, V in {h} x {d // h} dims; softmax(QK^T/sqrt(d_k))V", (b, t, d)),
        ("x + attention (residual)", "add the sub-layer output back", (b, t, d)),
        ("LayerNorm 2", "normalise before the FFN", (b, t, d)),
        ("Feed-forward network", f"{d} -> {cfg.ffn_mult * d} -> {d} ({cfg.activation})", (b, t, d)),
        ("x + FFN (residual)", "block output", (b, t, d)),
    ]


def dataflow_figure(stages: Sequence[tuple[str, str, tuple]], title: str = "Data flow") -> go.Figure:
    """Top-to-bottom flow chart: one box per stage, labelled with its tensor shape."""
    if not stages:
        return empty_figure("no stages")
    n = len(stages)
    step_y, box_h = 1.0, 0.62
    fig = go.Figure()
    for i, (name, desc, shape) in enumerate(stages):
        y = (n - 1 - i) * step_y
        fig.add_shape(
            type="rect", x0=0.05, x1=0.95, y0=y - box_h / 2, y1=y + box_h / 2,
            line=dict(color="#3b82d4"), fillcolor="#eef4fc",
        )
        fig.add_annotation(
            x=0.5, y=y, showarrow=False, align="center",
            text=f"<b>{name}</b><br><span style='font-size:11px'>{desc}</span><br>"
                 f"<span style='font-size:11px'>shape {tuple(shape)}</span>",
        )
        if i < n - 1:
            fig.add_annotation(
                x=0.5, y=y - box_h / 2 - 0.02, ax=0.5, ay=y - step_y + box_h / 2 + 0.02,
                xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                arrowhead=2, arrowwidth=1.5, arrowcolor="#57606a", text="",
            )
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[-0.6, n - 0.4])
    fig.update_layout(title=title, height=95 * n + 60, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def parameter_breakdown_figure(breakdown: dict[str, int]) -> go.Figure:
    """Horizontal bars of parameters per component (the ``total`` entry is excluded)."""
    items = [(k, v) for k, v in breakdown.items() if k != "total" and v > 0]
    if not items:
        return empty_figure("no parameters")
    items.sort(key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k.replace("_", " ") for k, _ in items], orientation="h"))
    fig.update_layout(
        title=f"Parameters by component (total {breakdown.get('total', sum(v for _, v in items)):,})",
        xaxis_title="parameters",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
