"""Tests for pages/page_transformer.py."""
from dataclasses import replace

from tests.conftest import assert_clean
from ui.state import ModelSettings

SMALL = ModelSettings(d_model=16, n_heads=2, n_layers=1, max_seq_len=16, use_input=False)
BPE = ModelSettings(tokenizer="bpe", bpe_merges=40, d_model=16, n_heads=2, n_layers=1, max_seq_len=16, use_input=False)


def test_renders_cleanly_with_a_single_layer(run_page):
    assert_clean(run_page("transformer", SMALL))


def test_shows_parameter_total(run_page):
    assert "Total parameters" in [m.label for m in run_page("transformer", SMALL).metric]


def test_long_text_is_cropped_with_warning(run_page):
    at = run_page("transformer", SMALL)
    at.text_input(key="tf_text").set_value("the robot read the book " * 5).run()
    assert_clean(at)
    assert any("context window" in w.value for w in at.warning)


def test_bpe_and_multiple_layers(run_page):
    at = run_page("transformer", replace(BPE, n_layers=3))
    at.slider(key="tf_layer").set_value(3).run()
    assert_clean(at)


def test_learned_positions_and_relu(run_page):
    at = run_page("transformer", replace(SMALL, positional="learned", activation="relu"))
    assert_clean(at)
