"""Tests for ui/lab.py: data preparation, training sessions, checkpoints."""
import pytest
import torch

from training.trainer import TrainConfig
from ui import lab
from ui.state import ModelSettings, apply_settings, init_state

SMALL = ModelSettings(d_model=16, n_heads=2, n_layers=1, max_seq_len=16, use_input=False)


@pytest.fixture
def state():
    s = {}
    init_state(s)
    assert apply_settings(s, SMALL) is None
    return s


def cfg(**kw):
    base = dict(steps=6, batch_size=4, block_size=8, eval_interval=3, eval_batches=2, warmup_steps=2, lr=3e-3)
    base.update(kw)
    return TrainConfig(**base)


# ----- data ------------------------------------------------------------------ #
def test_prepare_data_splits(state):
    train_data, val_data = lab.prepare_data(state)
    assert train_data.dtype == torch.long and val_data is not None
    assert len(train_data) > len(val_data) > 0


def test_prepare_data_too_little_text(state, monkeypatch):
    monkeypatch.setattr(lab, "load_corpus", lambda settings: "abc")
    with pytest.raises(ValueError, match="fewer than the context"):
        lab.prepare_data(state)


def test_prepare_data_without_validation_when_tiny(state, monkeypatch):
    monkeypatch.setattr(lab, "load_corpus", lambda settings: "abcdefghij" * 3)  # ~30 tokens
    train_data, val_data = lab.prepare_data(state, val_fraction=0.1)
    assert val_data is None and len(train_data) >= 18


def test_check_train_config(state):
    assert lab.check_train_config(state, cfg(block_size=8)) == []
    assert "larger than the model context" in lab.check_train_config(state, cfg(block_size=64))[0]


def test_schedule_preview_shape():
    schedule = lab.schedule_preview(cfg(steps=10, warmup_steps=4))
    assert schedule(0) < schedule(3) and schedule(9) < schedule(4)


# ----- training --------------------------------------------------------------- #
def test_run_training_records_history(state):
    steps = list(lab.run_training(state, cfg()))
    assert [m["step"] for m in steps] == list(range(1, 7))
    assert state["history"] == steps and steps[-1]["done"]
    assert state["training"] is False and state["model_source"] == "trained"
    assert not state["model"].training  # left in eval mode


def test_run_training_reduces_loss(state):
    list(lab.run_training(state, cfg(steps=40, lr=1e-2)))
    h = state["history"]
    assert sum(x["train_loss"] for x in h[-5:]) < sum(x["train_loss"] for x in h[:5])


def test_run_training_honours_stop_flag(state):
    seen = []
    for m in lab.run_training(state, cfg(steps=50)):
        seen.append(m)
        if len(seen) == 3:
            state["stop_requested"] = True
    assert len(state["history"]) == 3 and state["training"] is False
    assert "Stopped" in lab.finish_message(state)


def test_resume_continues_step_count(state):
    list(lab.run_training(state, cfg(steps=4)))
    list(lab.run_training(state, cfg(steps=4), resume=True))
    assert [m["step"] for m in state["history"]] == list(range(1, 9))


def test_fresh_run_resets_history(state):
    list(lab.run_training(state, cfg(steps=3)))
    list(lab.run_training(state, cfg(steps=2)))
    assert len(state["history"]) == 2


def test_training_flag_cleared_on_error(state):
    with pytest.raises(ValueError):
        list(lab.run_training(state, cfg(block_size=64)))
    assert state["training"] is False


def test_divergence_raises_and_clears_flag(state):
    with pytest.raises(FloatingPointError):
        list(lab.run_training(state, cfg(steps=30, optimizer="sgd", lr=1e9, grad_clip=0.0)))
    assert state["training"] is False


def test_finish_message_without_history(state):
    assert lab.finish_message(state) == "No steps were run."


def test_finish_message_reports_losses(state):
    list(lab.run_training(state, cfg()))
    msg = lab.finish_message(state)
    assert "Finished after 6 steps" in msg and "->" in msg


def test_history_summary(state):
    assert lab.history_summary([]) == {}
    list(lab.run_training(state, cfg()))
    s = lab.history_summary(state["history"])
    assert s["step"] == 6 and s["val_loss"] is not None and s["best_val_loss"] <= s["val_loss"]


# ----- checkpoints ------------------------------------------------------------ #
def test_save_and_load_round_trip(state, tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    list(lab.run_training(state, cfg()))
    path = lab.save_current(state, "demo run")
    assert path.parent == tmp_path and path.name.endswith("_demo_run.pt")
    assert lab.available_checkpoints() == [path]

    other = {}
    init_state(other)
    assert lab.load_into_state(other, path) is None
    assert other["model_source"] == "checkpoint" and len(other["history"]) == 6
    ids = torch.tensor([[1, 2, 3]])
    state["model"].eval()
    assert torch.allclose(state["model"](ids), other["model"](ids), atol=1e-6)


def test_save_without_model_fails():
    s = {}
    init_state(s)
    with pytest.raises(ValueError):
        lab.save_current(s)


def test_load_corrupted_checkpoint_reports_error(state, tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    old_model = state["model"]
    message = lab.load_into_state(state, bad)
    assert "cannot read checkpoint" in message
    assert state["model"] is old_model and state["last_error"] == message


def test_load_missing_checkpoint(state, tmp_path):
    assert "not found" in lab.load_into_state(state, tmp_path / "nope.pt")
