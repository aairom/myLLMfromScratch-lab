"""Unit tests for viz/training.py."""
import math

import plotly.graph_objects as go
import pytest
import torch

from llm.transformer import GPT, GPTConfig
from viz import training as vtr


def is_empty_message(fig):
    return isinstance(fig, go.Figure) and len(fig.data) == 0 and len(fig.layout.annotations) == 1


HISTORY = [
    {"step": i + 1, "train_loss": 3.0 - 0.1 * i, "val_loss": (2.9 - 0.1 * i) if i % 2 == 0 else None,
     "lr": 1e-3, "grad_norm": 0.5, "tokens_per_sec": 1000.0}
    for i in range(6)
]


def test_moving_average():
    assert vtr.moving_average([1, 2, 3, 4], 2) == [1.0, 1.5, 2.5, 3.5]
    assert vtr.moving_average([], 3) == []
    with pytest.raises(ValueError):
        vtr.moving_average([1], 0)


def test_loss_figure_traces():
    plain = vtr.loss_figure(HISTORY)
    assert [t.name for t in plain.data] == ["train (raw)", "validation"]
    assert list(plain.data[1].x) == [1, 3, 5]  # None validation entries skipped
    smooth = vtr.loss_figure(HISTORY, smooth_window=3, baseline=math.log(50), log_y=True)
    assert len(smooth.data) == 3 and smooth.layout.yaxis.type == "log"
    assert len(smooth.layout.shapes) == 1  # baseline line


def test_loss_figure_ignores_nan_and_empty():
    assert is_empty_message(vtr.loss_figure([]))
    assert is_empty_message(vtr.loss_figure([{"step": 1, "train_loss": float("nan")}]))


def test_metric_line_figures():
    for builder in (vtr.lr_figure, vtr.grad_norm_figure, vtr.throughput_figure):
        assert len(builder(HISTORY).data) == 1
        assert is_empty_message(builder([]))


def test_schedule_figure():
    fig = vtr.schedule_figure(lambda s: s * 0.1, 5)
    assert list(fig.data[0].y) == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert is_empty_message(vtr.schedule_figure(lambda s: 0, 0))


def test_model_and_block_stages_and_flow_figure():
    cfg = GPTConfig(vocab_size=50, d_model=16, n_heads=4, n_layers=3, max_seq_len=10)
    stages = vtr.model_stages(cfg, batch=2)
    assert stages[0][2] == (2, 10) and stages[-1][2] == (2, 10, 50)
    assert "x 3" in stages[3][0]
    block = vtr.block_stages(cfg, seq=7)
    assert block[2][2] == (1, 7, 16) and "4 heads" in block[2][0]
    fig = vtr.dataflow_figure(stages)
    assert len(fig.layout.shapes) == len(stages) and len(fig.layout.annotations) == 2 * len(stages) - 1
    assert is_empty_message(vtr.dataflow_figure([]))


def test_stage_shapes_match_real_model():
    cfg = GPTConfig(vocab_size=30, d_model=16, n_heads=2, n_layers=1, max_seq_len=8)
    model = GPT(cfg)
    ids = torch.randint(0, 30, (2, 6))
    _, details = model(ids, return_details=True)
    stages = vtr.model_stages(cfg, batch=2, seq=6)
    assert tuple(details["token_embeddings"].shape) == stages[1][2]
    assert tuple(details["final_norm"].shape) == stages[4][2]
    assert tuple(details["logits"].shape) == stages[5][2]


def test_parameter_breakdown_figure():
    model = GPT(GPTConfig(vocab_size=20, d_model=16, n_heads=2, n_layers=2, max_seq_len=8))
    bd = model.parameter_breakdown()
    fig = vtr.parameter_breakdown_figure(bd)
    assert sum(fig.data[0].x) == bd["total"]
    assert "total" not in list(fig.data[0].y)
    assert is_empty_message(vtr.parameter_breakdown_figure({"total": 0}))
