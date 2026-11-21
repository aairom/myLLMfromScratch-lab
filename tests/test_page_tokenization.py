"""Tests for pages/page_tokenization.py."""
from tests.conftest import assert_clean
from ui.state import ModelSettings

SMALL = ModelSettings(d_model=16, n_heads=2, n_layers=1, max_seq_len=16, use_input=False)
BPE = ModelSettings(tokenizer="bpe", bpe_merges=40, d_model=16, n_heads=2, n_layers=1, max_seq_len=16, use_input=False)


def test_renders_cleanly(run_page):
    assert_clean(run_page("tokenization", SMALL))


def test_shows_token_chips_and_metrics(run_page):
    at = run_page("tokenization", SMALL)
    assert any('title="id' in m.value for m in at.markdown)
    assert "Characters per token" in [m.label for m in at.metric]


def test_bpe_replay(run_page):
    at = run_page("tokenization", BPE)
    assert_clean(at)
    at.slider(key="tok_upto").set_value(0).run()
    assert_clean(at)


def test_char_tokenizer_hints_at_bpe(run_page):
    assert any("bpe" in i.value for i in run_page("tokenization", SMALL).info)


def test_empty_text(run_page):
    at = run_page("tokenization", SMALL)
    at.text_area(key="tok_text").set_value("").run()
    assert_clean(at)
    assert any("Type some text" in i.value for i in at.info)


def test_unknown_character_warning(run_page):
    at = run_page("tokenization", SMALL)
    at.text_area(key="tok_text").set_value("robot \u2603").run()
    assert any("<unk>" in w.value for w in at.warning)
