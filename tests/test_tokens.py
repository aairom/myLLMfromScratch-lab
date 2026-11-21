"""Unit tests for viz/tokens.py."""
import plotly.graph_objects as go

from llm.tokenizer import BPETokenizer, CharTokenizer
from viz import tokens as vt


def is_empty_message(fig):
    return isinstance(fig, go.Figure) and len(fig.data) == 0 and len(fig.layout.annotations) == 1


def test_tokens_html_escapes_and_shows_ids():
    html = vt.tokens_html(["<b>", " x"], [3, 4])
    assert "&lt;b&gt;" in html and 'title="id 3"' in html and "␣x" in html
    assert "no tokens" in vt.tokens_html([], [])


def test_token_rows():
    rows = vt.token_rows(["a", " b"], [1, 2])
    assert rows == [{"position": 0, "token": "a", "id": 1}, {"position": 1, "token": "␣b", "id": 2}]


def test_merge_rows_from_bpe_trace():
    tok = BPETokenizer.train("low low lower lowest", 4)
    rows = vt.merge_rows(tok)
    assert len(rows) == len(tok.trace) == 4
    assert rows[0]["step"] == 1 and rows[0]["count"] >= rows[-1]["count"] - 100
    assert vt.merge_rows(CharTokenizer.train("abc")) == []  # no trace


def test_vocab_frequency_and_compression_figures():
    fig = vt.vocab_frequency_figure(list("aabbbc"), top_n=2)
    assert list(fig.data[0].y) == [3, 2]
    assert is_empty_message(vt.vocab_frequency_figure([]))
    comp = vt.compression_figure(100, {0: 100, 5: 70, 10: 55})
    assert list(comp.data[0].x) == [0, 5, 10]
    assert is_empty_message(vt.compression_figure(100, {}))
