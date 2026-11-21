"""Tests for pages/page_training.py (the training lab)."""
import pytest

from tests.conftest import assert_clean
from ui.state import ModelSettings

SMALL = ModelSettings(d_model=16, n_heads=2, n_layers=1, max_seq_len=16, use_input=False)


def start_training(at, steps=12):
    at.number_input(key="tr_steps").set_value(steps)
    at.number_input(key="tr_block").set_value(8)
    at.number_input(key="tr_redraw").set_value(4)
    at.number_input(key="tr_warm").set_value(2)
    at.run()
    at.button(key="tr_start").click().run()
    return at



def test_training_page_initial_state(run_page):
    at = run_page("training", SMALL)
    assert_clean(at)
    assert any("No training run yet" in i.value for i in at.info)
    assert at.button(key="tr_save").disabled  # nothing to save yet
    assert at.button(key="tr_resume").disabled


def test_training_run_streams_metrics_and_fills_history(run_page):
    at = start_training(run_page("training", SMALL))
    assert_clean(at)
    history = at.session_state["history"]
    assert len(history) == 12 and history[-1]["done"]
    assert at.session_state["model_source"] == "trained"
    assert {"Step", "Train loss", "Val loss", "Learning rate", "Tokens / s"} <= {m.label for m in at.metric}


def test_training_reduces_loss_on_the_page(run_page):
    at = start_training(run_page("training", SMALL), steps=40)
    h = at.session_state["history"]
    assert h[-1]["train_loss"] < h[0]["train_loss"]


def test_training_resume_continues_steps(run_page):
    at = start_training(run_page("training", SMALL), steps=6)
    at.run()
    at.button(key="tr_resume").click().run()
    assert_clean(at)
    assert [m["step"] for m in at.session_state["history"]] == list(range(1, 13))


def test_training_window_larger_than_context_blocks_start(run_page):
    at = run_page("training", SMALL)
    at.number_input(key="tr_block").set_value(100).run()
    assert any("larger than the model context" in e.value for e in at.error)
    assert at.button(key="tr_start").disabled


def test_training_diverging_run_shows_message_not_crash(run_page):
    at = run_page("training", SMALL)
    at.selectbox(key="tr_opt").set_value("sgd")
    at.number_input(key="tr_lr").set_value(1.0)
    at.number_input(key="tr_clip").set_value(0.0)
    at.number_input(key="tr_steps").set_value(300)
    at.number_input(key="tr_block").set_value(8).run()
    at.button(key="tr_start").click().run()
    assert not at.exception  # either it trains or reports divergence, but never crashes


@pytest.mark.parametrize("optimizer", ["adam", "adamw", "sgd", "torch_adamw", "torch_sgd"])
def test_training_every_optimizer(run_page, optimizer):
    at = run_page("training", SMALL)
    at.selectbox(key="tr_opt").set_value(optimizer)
    at.number_input(key="tr_lr").set_value(0.003)
    at.number_input(key="tr_steps").set_value(4)
    at.number_input(key="tr_block").set_value(8).run()
    at.button(key="tr_start").click().run()
    assert_clean(at)
    assert len(at.session_state["history"]) == 4


def test_training_constant_schedule(run_page):
    at = run_page("training", SMALL)
    at.selectbox(key="tr_sched").set_value("constant")
    at.number_input(key="tr_steps").set_value(3)
    at.number_input(key="tr_block").set_value(8).run()
    at.button(key="tr_start").click().run()
    assert_clean(at)
    assert len({h["lr"] for h in at.session_state["history"]}) == 1


def test_training_too_little_text_is_reported(run_page, monkeypatch):
    import ui.lab as lab

    monkeypatch.setattr(lab, "load_corpus", lambda settings: "abc")
    at = run_page("training", SMALL)
    at.number_input(key="tr_block").set_value(8).run()
    at.button(key="tr_start").click().run()
    assert not at.exception
    assert any("fewer than the context" in e.value for e in at.error)


def test_checkpoint_save_and_load_through_the_page(run_page, tmp_path):
    at = start_training(run_page("training", SMALL), steps=5)
    at.run()
    at.button(key="tr_save").click().run()
    assert any(s.value.startswith("Saved ") for s in at.success)
    assert len(list((tmp_path / "output").glob("*.pt"))) == 1
    at.button(key="tr_load").click().run()
    assert not at.exception
    assert at.session_state["model_source"] == "checkpoint"


def test_corrupted_checkpoint_shows_error(run_page, tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "20990101_000000_broken.pt").write_bytes(b"garbage")
    at = run_page("training", SMALL)
    at.button(key="tr_load").click().run()
    assert not at.exception
    assert any("cannot read checkpoint" in e.value for e in at.error)
