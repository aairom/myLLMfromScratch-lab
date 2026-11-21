"""Tests for pages/page_multihead.py."""
from tests.conftest import assert_clean
from ui.state import ModelSettings

SMALL = ModelSettings(d_model=16, n_heads=2, n_layers=1, max_seq_len=16, use_input=False)


def test_renders_cleanly(run_page):
    assert_clean(run_page("multihead", SMALL))


def test_invalid_head_count_shows_message(run_page):
    at = run_page("multihead", SMALL)
    at.number_input(key="mh_heads").set_value(5).run()
    assert not at.exception
    assert any("not divisible" in e.value for e in at.error)


def test_valid_head_counts_including_a_single_head(run_page):
    at = run_page("multihead", SMALL)
    for heads in (1, 2, 8):
        at.number_input(key="mh_heads").set_value(heads).run()
        assert_clean(at)
