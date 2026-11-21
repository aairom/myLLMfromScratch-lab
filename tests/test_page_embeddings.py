"""Tests for pages/page_embeddings.py."""
import pytest

from tests.conftest import assert_clean
from ui.state import ModelSettings

SMALL = ModelSettings(d_model=16, n_heads=2, n_layers=1, max_seq_len=16, use_input=False)


@pytest.mark.parametrize("init", ["normal", "uniform", "xavier"])
def test_every_init(run_page, init):
    at = run_page("embeddings", SMALL)
    at.selectbox(key="emb_init").set_value(init).run()
    assert_clean(at)


def test_scale_toggle(run_page):
    at = run_page("embeddings", SMALL)
    at.toggle(key="emb_scale").set_value(True).run()
    assert_clean(at)


def test_empty_text(run_page):
    at = run_page("embeddings", SMALL)
    at.text_input(key="emb_text").set_value("").run()
    assert_clean(at)
