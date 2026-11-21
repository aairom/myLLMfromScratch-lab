"""Tests for pages/page_attention.py."""
import pytest

from tests.conftest import assert_clean
from ui.state import ModelSettings

SMALL = ModelSettings(d_model=16, n_heads=2, n_layers=1, max_seq_len=16, use_input=False)


@pytest.mark.parametrize("step", ["1. Q, K, V", "2. Scores", "3. Softmax", "4. Output"])
def test_every_stage(run_page, step):
    at = run_page("attention", SMALL)
    at.segmented_control(key="att_step").set_value(step).run()
    assert_clean(at)


def test_without_causal_mask(run_page):
    at = run_page("attention", SMALL)
    at.toggle(key="att_causal").set_value(False).run()
    assert_clean(at)


def test_empty_text_is_handled(run_page):
    at = run_page("attention", SMALL)
    at.text_input(key="att_text").set_value("").run()
    assert not at.exception
    assert any("Cannot run attention" in i.value for i in at.info)


def test_single_token_text(run_page):
    """One token means one row: the row slider must not crash."""
    at = run_page("attention", SMALL)
    at.text_input(key="att_text").set_value("a").run()
    at.segmented_control(key="att_step").set_value("3. Softmax").run()
    assert_clean(at)
