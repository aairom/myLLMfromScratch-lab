"""Tests for ui/state.py: validation, model building and session-state transitions."""
import pytest
import torch

from llm.tokenizer import build_tokenizer
from llm.transformer import GPT, GPTConfig
from training.checkpoint import load_checkpoint, save_checkpoint
from ui import state as st_mod
from ui.state import (
    ModelSettings,
    adopt_checkpoint,
    apply_settings,
    build_model,
    default_settings,
    ensure_model,
    init_state,
    settings_from_model,
)


@pytest.fixture
def small():
    """Tiny settings that build in milliseconds and ignore the input/ folder."""
    return ModelSettings(d_model=16, n_heads=2, n_layers=1, max_seq_len=16, use_input=False)


# ----- validation ---------------------------------------------------------- #
def test_default_settings_are_valid():
    assert default_settings().problems() == []


def test_heads_must_divide_d_model(small):
    bad = ModelSettings(d_model=30, n_heads=4)
    problems = bad.problems()
    assert any("divisible" in p for p in problems)
    assert small.problems() == []


def test_multiple_problems_are_all_reported():
    bad = ModelSettings(tokenizer="word", d_model=0, dropout=1.5, use_bundled=False, use_input=False)
    assert len(bad.problems()) >= 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("activation", "swish"),
        ("positional", "rope"),
        ("embedding_init", "zeros"),
        ("bpe_merges", -1),
        ("bpe_merges", 10_000),
        ("n_layers", 0),
    ],
)
def test_invalid_fields_are_rejected(small, field, value):
    from dataclasses import replace

    assert replace(small, **{field: value}).problems()


def test_gpt_config_uses_settings(small):
    cfg = small.gpt_config(vocab_size=50)
    assert isinstance(cfg, GPTConfig)
    assert (cfg.vocab_size, cfg.d_model, cfg.n_heads) == (50, 16, 2)


def test_to_dict_round_trips_fields(small):
    assert ModelSettings(**small.to_dict()) == small


# ----- building ------------------------------------------------------------- #
def test_build_model_returns_matching_vocab(small):
    model, tok = build_model(small)
    assert isinstance(model, GPT)
    assert model.cfg.vocab_size == tok.vocab_size


def test_build_model_is_deterministic_for_a_seed(small):
    m1, _ = build_model(small)
    m2, _ = build_model(small)
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_build_model_raises_readable_error():
    with pytest.raises(ValueError, match="divisible"):
        build_model(ModelSettings(d_model=30, n_heads=4))


def test_bpe_tokenizer_grows_vocab(small):
    from dataclasses import replace

    _, char_tok = build_model(small)
    _, bpe_tok = build_model(replace(small, tokenizer="bpe", bpe_merges=20))
    assert bpe_tok.vocab_size == char_tok.vocab_size + 20


# ----- session state ------------------------------------------------------- #
def test_init_state_fills_defaults_once():
    state = {}
    init_state(state)
    assert state["model"] is None and state["history"] == []
    state["history"].append({"step": 1})
    init_state(state)  # must not overwrite
    assert state["history"] == [{"step": 1}]


def test_init_state_gives_independent_lists():
    a, b = {}, {}
    init_state(a)
    init_state(b)
    a["history"].append(1)
    assert b["history"] == []


def test_apply_settings_success_resets_progress(small):
    state = {}
    init_state(state)
    state["history"] = [{"step": 1}]
    assert apply_settings(state, small) is None
    assert state["model"] is not None and state["history"] == []
    assert state["model_source"] == "fresh" and state["settings"] == small


def test_apply_settings_failure_keeps_old_model(small):
    state = {}
    init_state(state)
    apply_settings(state, small)
    old = state["model"]
    msg = apply_settings(state, ModelSettings(d_model=30, n_heads=4))
    assert "divisible" in msg
    assert state["model"] is old and state["last_error"] == msg


def test_ensure_model_builds_only_when_missing(small):
    state = {}
    init_state(state)
    state["settings"] = small
    assert ensure_model(state) is None
    first = state["model"]
    assert ensure_model(state) is None
    assert state["model"] is first


def test_ensure_model_reports_error_without_data():
    state = {}
    init_state(state)
    state["settings"] = ModelSettings(use_bundled=False, use_input=False)
    assert ensure_model(state)
    assert state["model"] is None


def test_missing_text_source_error(monkeypatch, small):
    from dataclasses import replace

    monkeypatch.setattr(st_mod, "build_corpus", lambda *a, **k: (_ for _ in ()).throw(ValueError("no text")))
    state = {}
    init_state(state)
    assert "no text" in apply_settings(state, replace(small))


# ----- checkpoints ---------------------------------------------------------- #
def test_adopt_checkpoint_matches_sidebar_to_model(small, tmp_path):
    model, tok = build_model(small)
    path = save_checkpoint(model, tok, history=[{"step": 1, "train_loss": 3.0}], path=tmp_path / "m.pt")
    state = {}
    init_state(state)
    adopt_checkpoint(state, load_checkpoint(path))
    assert state["model_source"] == "checkpoint"
    assert state["settings"].d_model == 16 and state["settings"].n_layers == 1
    assert state["history"] == [{"step": 1, "train_loss": 3.0}]
    assert state["optimizer"] is None


def test_settings_from_model_for_bpe():
    tok = build_tokenizer("bpe", "abab abab ab", 3)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, d_model=8, n_heads=2, n_layers=1, max_seq_len=8))
    s = settings_from_model(model, tok, ModelSettings())
    assert s.tokenizer == "bpe" and s.bpe_merges == 3 and s.d_model == 8


def test_get_device_returns_torch_device():
    assert isinstance(st_mod.get_device({}), torch.device)
