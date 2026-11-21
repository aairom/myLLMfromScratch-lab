"""Accessible inline-SVG line charts for the documentation (no external assets).

The charts draw the *reference run* stored in ``docsgen/reference_run.json``: the
300-step AdamW training run of the default model on the bundled corpus, exactly
as the Training lab page runs it. Each chart carries a ``<title>``, a ``<desc>``
and a collapsible data table, so the information is not locked inside a picture.
"""
from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

REFERENCE_PATH = Path(__file__).resolve().parent / "reference_run.json"

WIDTH, HEIGHT = 720, 340
MARGIN = dict(left=64, right=24, top=18, bottom=52)


@dataclass
class Series:
    """One polyline (or a set of markers when ``markers`` is true)."""

    name: str
    xs: Sequence[float]
    ys: Sequence[float]
    css: str = "s0"  # CSS class that decides colour and dash pattern
    markers: bool = False


@dataclass
class Chart:
    chart_id: str
    title: str
    description: str
    x_label: str
    y_label: str
    series: list = field(default_factory=list)
    hlines: list = field(default_factory=list)  # (value, label)
    y_digits: int = 2


def load_reference(path: Path = REFERENCE_PATH) -> dict:
    """The stored reference run (config plus per-step metrics)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Round tick values (1, 2 or 5 times a power of ten) covering ``[lo, hi]``."""
    if not hi > lo:
        return [lo]
    raw = (hi - lo) / max(1, target)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw)
    first = math.ceil(lo / step) * step
    ticks, value = [], first
    while value <= hi + step * 1e-9:
        ticks.append(round(value, 12))
        value += step
    return ticks


def moving_average(values: Sequence[float], window: int) -> list[float]:
    """Trailing mean (same definition as ``viz.training.moving_average``)."""
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _scale(value: float, lo: float, hi: float, a: float, b: float) -> float:
    return a + (value - lo) / (hi - lo) * (b - a) if hi > lo else (a + b) / 2


def render_chart(chart: Chart) -> str:
    """Return a ``<figure>`` with the SVG, a caption and the data table."""
    xs_all = [x for s in chart.series for x in s.xs]
    ys_all = [y for s in chart.series for y in s.ys] + [v for v, _ in chart.hlines]
    if not xs_all:
        raise ValueError(f"chart {chart.chart_id!r} has no data")
    x_lo, x_hi = min(xs_all), max(xs_all)
    y_ticks = nice_ticks(min(ys_all), max(ys_all))
    y_lo = min(min(ys_all), y_ticks[0])
    y_hi = max(max(ys_all), y_ticks[-1])
    pad = (y_hi - y_lo) * 0.04 or 1.0
    y_lo, y_hi = y_lo - pad, y_hi + pad
    x_ticks = nice_ticks(x_lo, x_hi, 6)

    left, right = MARGIN["left"], WIDTH - MARGIN["right"]
    top, bottom = MARGIN["top"], HEIGHT - MARGIN["bottom"]

    def px(x: float) -> float:
        return _scale(x, x_lo, x_hi, left, right)

    def py(y: float) -> float:
        return _scale(y, y_lo, y_hi, bottom, top)

    parts = [
        f'<svg class="chart-svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-labelledby="{chart.chart_id}-t {chart.chart_id}-d" xmlns="http://www.w3.org/2000/svg">',
        f'<title id="{chart.chart_id}-t">{html.escape(chart.title)}</title>',
        f'<desc id="{chart.chart_id}-d">{html.escape(chart.description)}</desc>',
        f'<rect class="plot-bg" x="{left}" y="{top}" width="{right - left}" height="{bottom - top}"/>',
    ]
    for tick in y_ticks:
        y = py(tick)
        parts.append(f'<line class="grid" x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:.{chart.y_digits}f}</text>')
    for tick in x_ticks:
        x = px(tick)
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{bottom}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{bottom + 18}" text-anchor="middle">{tick:g}</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{bottom}"/>')
    parts.append(f'<text class="label" x="{(left + right) / 2:.0f}" y="{HEIGHT - 8}" text-anchor="middle">{html.escape(chart.x_label)}</text>')
    parts.append(
        f'<text class="label" transform="translate(16 {(top + bottom) / 2:.0f}) rotate(-90)" '
        f'text-anchor="middle">{html.escape(chart.y_label)}</text>'
    )
    for value, label in chart.hlines:
        y = py(value)
        parts.append(f'<line class="ref" x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{right - 4}" y="{y - 5:.1f}" text-anchor="end">{html.escape(label)}</text>')
    for s in chart.series:
        points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(s.xs, s.ys))
        if not s.markers:
            parts.append(f'<polyline class="line {s.css}" points="{points}"/>')
        else:
            parts.append(f'<polyline class="line {s.css}" points="{points}"/>')
            for x, y in zip(s.xs, s.ys):
                parts.append(f'<circle class="dot {s.css}" cx="{px(x):.1f}" cy="{py(y):.1f}" r="3"/>')
    parts.append("</svg>")

    legend = "".join(
        f'<li><svg width="30" height="10" aria-hidden="true"><line class="line {s.css}" x1="0" x2="30" y1="5" y2="5"/></svg> '
        f"{html.escape(s.name)}</li>"
        for s in chart.series
    )
    if chart.hlines:
        legend += "".join(
            f'<li><svg width="30" height="10" aria-hidden="true"><line class="ref" x1="0" x2="30" y1="5" y2="5"/></svg> '
            f"{html.escape(label)}</li>" for _, label in chart.hlines
        )
    return (
        f'<figure class="chart" id="{chart.chart_id}">'
        f'<figcaption><strong>{html.escape(chart.title)}.</strong> {html.escape(chart.description)}</figcaption>'
        + "".join(parts)
        + f'<ul class="legend">{legend}</ul>{_data_table(chart)}</figure>'
    )


def _data_table(chart: Chart, max_rows: int = 13) -> str:
    """A small table sampled from the first series, for readers who cannot see the plot."""
    base = chart.series[0]
    n = len(base.xs)
    stride = max(1, math.ceil(n / max_rows))
    picks = sorted(set(list(range(0, n, stride)) + [n - 1]))
    head = "".join(f"<th scope=\"col\">{html.escape(s.name)}</th>" for s in chart.series)
    rows = []
    for i in picks:
        x = base.xs[i]
        cells = []
        for s in chart.series:
            lookup = dict(zip(s.xs, s.ys))
            cells.append(f"<td>{lookup[x]:.{chart.y_digits + 1}f}</td>" if x in lookup else "<td>-</td>")
        rows.append(f"<tr><th scope=\"row\">{x:g}</th>{''.join(cells)}</tr>")
    return (
        '<details class="data"><summary>Show the data behind this chart</summary><div class="scroll" tabindex="0">'
        f'<table><caption>{html.escape(chart.title)} (sampled steps)</caption>'
        f'<thead><tr><th scope="col">{html.escape(chart.x_label)}</th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></details>'
    )


# --------------------------------------------------------------------------- #
# The three charts used by the guides
# --------------------------------------------------------------------------- #
def _history(run: dict) -> list[dict]:
    return run["history"]


def loss_chart(run: Optional[dict] = None, vocab_size: int = 51) -> str:
    """Training loss (raw and smoothed), validation loss and the ``ln(V)`` baseline."""
    run = run or load_reference()
    hist = _history(run)
    steps = [h["step"] for h in hist]
    train = [h["train_loss"] for h in hist]
    val = [(h["step"], h["val_loss"]) for h in hist if h.get("val_loss") is not None]
    chart = Chart(
        "chart-loss", "Cross-entropy loss during the reference run",
        f"Training loss falls from {train[0]:.3f} to {train[-1]:.3f} in {steps[-1]} steps; "
        f"validation loss ends at {val[-1][1]:.3f}. The dotted line is the loss of a uniform guesser, ln({vocab_size}).",
        "step", "loss (nats per token)",
        series=[
            Series("train (raw)", steps, train, "s0 thin"),
            Series("train (average of 15)", steps, moving_average(train, 15), "s1"),
            Series("validation", [s for s, _ in val], [v for _, v in val], "s2", markers=True),
        ],
        hlines=[(math.log(vocab_size), f"uniform guess ln({vocab_size}) = {math.log(vocab_size):.3f}")],
        y_digits=2,
    )
    return render_chart(chart)


def lr_chart(run: Optional[dict] = None) -> str:
    """Learning rate per step: linear warm-up, then cosine decay."""
    run = run or load_reference()
    hist = _history(run)
    cfg = run["config"]
    chart = Chart(
        "chart-lr", "Learning-rate schedule of the reference run",
        f"Linear warm-up over {cfg['warmup_steps']} steps to {cfg['lr']:g}, then cosine decay "
        f"to {cfg['lr'] * cfg['min_lr_ratio']:g} at step {cfg['steps']}.",
        "step", "learning rate",
        series=[Series("learning rate", [h["step"] for h in hist], [h["lr"] for h in hist], "s1")],
        y_digits=4,
    )
    return render_chart(chart)


def grad_norm_chart(run: Optional[dict] = None) -> str:
    """Global gradient norm before clipping."""
    run = run or load_reference()
    hist = _history(run)
    clip = run["config"]["grad_clip"]
    chart = Chart(
        "chart-gradnorm", "Gradient norm of the reference run",
        f"Global L2 norm of the gradient before clipping; the clipping threshold is {clip:g}.",
        "step", "gradient norm",
        series=[Series("gradient norm", [h["step"] for h in hist], [h["grad_norm"] for h in hist], "s0")],
        hlines=[(clip, f"clip threshold {clip:g}")],
        y_digits=2,
    )
    return render_chart(chart)


CHARTS = {"loss": loss_chart, "lr": lr_chart, "gradnorm": grad_norm_chart}
