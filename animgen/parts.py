"""Small drawing helpers used by several scenes (token rows, wrapped text, fades)."""
from __future__ import annotations

from typing import Optional, Sequence

from animgen import fonts, theme
from animgen.canvas import Canvas
from animgen.theme import RGB


def fade(colour: RGB, amount: float) -> RGB:
    """``colour`` blended towards the background (``amount`` 1 = full colour, 0 = invisible)."""
    return theme.mix(theme.BG, colour, amount)


def wrap(c: Canvas, text: str, width: float, size: float = 12, kind: str = fonts.SANS) -> list[str]:
    """Greedy word wrap of ``text`` to ``width`` design units."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and c.text_width(trial, size, kind) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def paragraph(c: Canvas, x: float, y: float, text: str, width: float, size: float = 12, kind: str = fonts.SANS,
              fill: RGB = theme.INK, line_h: Optional[float] = None, anchor: str = "la") -> float:
    """Draw wrapped text and return the y below the last line."""
    step = line_h or size * 1.35
    for line in wrap(c, text, width, size, kind):
        c.text(x, y, line, size, kind, fill, anchor)
        y += step
    return y


def chip_row(c: Canvas, x: float, y: float, labels: Sequence[str], fill: RGB | Sequence[RGB] = theme.TOKEN,
             size: float = 14, height: float = 28, gap: float = 3, min_w: float = 26, shown: Optional[float] = None,
             kind: str = fonts.MONO, text_fill: RGB = theme.PANEL) -> list[tuple[float, float]]:
    """A row of token chips. ``shown`` (a float) reveals chips one by one; the next one fades in.

    Returns ``[(x, width), ...]`` of every chip (also the hidden ones) so callers can attach arrows.
    """
    boxes: list[tuple[float, float]] = []
    cx = x
    for i, label in enumerate(labels):
        w = max(min_w, c.text_width(fonts.visible_token(label), size, kind) + 14)
        boxes.append((cx, w))
        amount = 1.0 if shown is None else theme.clamp01(shown - i)
        if amount > 0:
            base = fill[i] if not isinstance(fill, tuple) else fill
            c.rect(cx, y, cx + w, y + height, fill=fade(base, amount), radius=6)
            c.text(cx + w / 2, y + height / 2, fonts.visible_token(label), size, kind, fade(text_fill, 0.4 + 0.6 * amount), "mm")
        cx += w + gap
    return boxes


def row_width(c: Canvas, labels: Sequence[str], size: float = 14, gap: float = 3, min_w: float = 26,
              kind: str = fonts.MONO) -> float:
    total = 0.0
    for label in labels:
        total += max(min_w, c.text_width(fonts.visible_token(label), size, kind) + 14) + gap
    return total - gap


def centred_x(c: Canvas, labels: Sequence[str], size: float = 14, gap: float = 3, min_w: float = 26,
              kind: str = fonts.MONO) -> float:
    """Left x that centres a chip row in the content area."""
    return (800 - row_width(c, labels, size, gap, min_w, kind)) / 2


def panel_title(c: Canvas, x: float, y: float, text: str, colour: RGB = theme.INK, size: float = 13) -> None:
    c.text(x, y, text, size, fonts.BOLD, colour)


def note(c: Canvas, x: float, y: float, text: str, size: float = 12, fill: RGB = theme.MUTED, anchor: str = "la") -> None:
    c.text(x, y, text, size, fonts.SANS, fill, anchor)


def check(c: Canvas, cx: float, cy: float, r: float = 8, colour: RGB = theme.VALUE, progress: float = 1.0) -> None:
    """A tick mark drawn with lines (the font may lack the glyph); ``progress`` draws it stroke by stroke."""
    if progress <= 0:
        return
    c.circle(cx, cy, r, fill=theme.tint(colour, 0.2), outline=colour, width=1.5)
    p0, p1, p2 = (cx - r * 0.45, cy), (cx - r * 0.1, cy + r * 0.4), (cx + r * 0.5, cy - r * 0.35)
    first = theme.segment(progress, 0.0, 0.5)
    second = theme.segment(progress, 0.5, 1.0)
    mid = (theme.lerp(p0[0], p1[0], first), theme.lerp(p0[1], p1[1], first))
    c.line([p0, mid], colour, 2.2)
    if second > 0:
        c.line([p1, (theme.lerp(p1[0], p2[0], second), theme.lerp(p1[1], p2[1], second))], colour, 2.2)


def bars(c: Canvas, x: float, y: float, labels: Sequence[str], values: Sequence[float], width: float = 200,
         row_h: float = 18, colour: RGB = theme.TOKEN, vmax: Optional[float] = None, reveal: float = 1.0,
         size: float = 12, fmt: str = ".3f", label_w: float = 34, highlight: Optional[int] = None,
         show_values: bool = True) -> None:
    """Horizontal bar chart; ``reveal`` grows the bars from zero."""
    top = vmax if vmax else (max(values) if values else 1.0) or 1.0
    for i, (label, v) in enumerate(zip(labels, values)):
        yy = y + i * row_h
        c.text(x + label_w - 6, yy + row_h / 2 - 1, fonts.visible_token(label), size, fonts.MONO, theme.INK, "rm")
        col = theme.QUERY if highlight == i else colour
        c.bar(x + label_w, yy + 3, width, row_h - 8, theme.clamp01(v / top) * reveal, col)
        if show_values:
            c.text(x + label_w + width + 6, yy + row_h / 2 - 1, format(v * reveal, fmt), size - 1,
                   fonts.MONO, theme.MUTED, "lm")
