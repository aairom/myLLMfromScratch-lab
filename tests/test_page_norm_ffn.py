"""Tests for pages/page_norm_ffn.py."""
import pytest

from tests.conftest import assert_clean
from ui.state import ModelSettings

SMALL = ModelSettings(d_model=16, n_heads=2, n_layers=1, max_seq_len=16, use_input=False)


@pytest.mark.parametrize("stage", ["mean", "variance", "normalized", "scaled output"])
def test_every_layernorm_stage(run_page, stage):
    at = run_page("norm_ffn", SMALL)
    at.segmented_control(key="ln_stage").set_value(stage).run()
    assert_clean(at)


@pytest.mark.parametrize("act", ["gelu", "relu"])
def test_activations(run_page, act):
    at = run_page("norm_ffn", SMALL)
    at.selectbox(key="ffn_act").set_value(act).run()
    assert_clean(at)


def test_empty_text_is_handled(run_page):
    at = run_page("norm_ffn", SMALL)
    at.text_input(key="ln_text").set_value("").run()
    assert not at.exception
