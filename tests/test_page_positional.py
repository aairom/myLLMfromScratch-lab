"""Tests for pages/page_positional.py."""
import pytest

from tests.conftest import assert_clean
from ui.state import ModelSettings

SMALL = ModelSettings(d_model=16, n_heads=2, n_layers=1, max_seq_len=16, use_input=False)


@pytest.mark.parametrize("kind", ["sinusoidal", "learned"])
def test_kinds(run_page, kind):
    at = run_page("positional", SMALL)
    at.selectbox(key="pos_kind").set_value(kind).run()
    assert_clean(at)


def test_odd_width(run_page):
    at = run_page("positional", SMALL)
    at.slider(key="pos_dim").set_value(7).run()
    assert_clean(at)
