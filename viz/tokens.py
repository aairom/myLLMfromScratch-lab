"""Figures for the tokenization page."""
from __future__ import annotations

from collections import Counter
from html import escape

import plotly.graph_objects as go

from .common import display_token, empty_figure

# Soft background colours cycled over consecutive tokens so boundaries are visible.
_PALETTE = ("#cfe8ff", "#ffe3c2", "#d6f5d6", "#f6d1ea", "#fff3b0", "#e0d7ff")


def tokens_html(tokens: list[str], ids: list[int]) -> str:
    """HTML for coloured token chips; hovering a chip shows its id."""
    if not tokens:
        return "<em>no tokens</em>"
    chips = []
    for i, (tok, tid) in enumerate(zip(tokens, ids)):
        colour = _PALETTE[i % len(_PALETTE)]
        label = escape(display_token(tok))
        chips.append(
            f'<span title="id {tid}" style="background:{colour};color:#1f2328;padding:2px 4px;'
            f'margin:1px;border-radius:4px;font-family:monospace;display:inline-block">{label}</span>'
        )
    return "".join(chips)


def token_rows(tokens: list[str], ids: list[int]) -> list[dict]:
    """Table rows ``position / token / id`` for the encoded text."""
    return [{"position": i, "token": display_token(t), "id": tid} for i, (t, tid) in enumerate(zip(tokens, ids))]


def merge_rows(tokenizer) -> list[dict]:
    """One row per BPE merge: what was merged, how often, and the runner-up pairs."""
    rows = []
    for step in getattr(tokenizer, "trace", []):
        runner_ups = ", ".join(
            f"{display_token(a)}+{display_token(b)} ({c})" for (a, b), c in step.top_pairs[1:]
        )
        rows.append(
            {
                "step": step.step,
                "merged pair": f"{display_token(step.pair[0])} + {display_token(step.pair[1])}",
                "count": step.count,
                "new token": display_token(step.new_token),
                "runner-ups": runner_ups,
            }
        )
    return rows


def vocab_frequency_figure(tokens: list[str], top_n: int = 25) -> go.Figure:
    """Bar chart of the ``top_n`` most frequent tokens in the encoded text."""
    if not tokens:
        return empty_figure("no tokens to count")
    counts = Counter(tokens).most_common(top_n)
    fig = go.Figure(go.Bar(x=[display_token(t) for t, _ in counts], y=[c for _, c in counts]))
    fig.update_layout(
        title=f"Most frequent tokens (top {len(counts)})",
        xaxis=dict(title="token", type="category"),
        yaxis_title="count",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def compression_figure(char_count: int, token_counts: dict[int, int]) -> go.Figure:
    """Sequence length versus number of BPE merges (more merges -> shorter sequences)."""
    if not token_counts:
        return empty_figure("no data")
    merges = sorted(token_counts)
    fig = go.Figure(go.Scatter(x=merges, y=[token_counts[m] for m in merges], mode="lines+markers", name="tokens"))
    fig.add_hline(y=char_count, line_dash="dot", annotation_text="characters (0 merges)")
    fig.update_layout(
        title="Tokens needed for the text vs. number of merges",
        xaxis_title="BPE merges",
        yaxis_title="sequence length (tokens)",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
