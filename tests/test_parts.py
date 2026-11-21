"""Unit tests for animgen/parts.py."""
import pytest

from animgen import fonts, parts, theme
from animgen.canvas import Canvas


@pytest.fixture
def c():
    return Canvas(800, 450, 1)


def px(c, x, y):
    return c.finish().getpixel((x, y))


def test_fade_blends_towards_the_background():
    assert parts.fade(theme.LOSS, 1) == theme.LOSS and parts.fade(theme.LOSS, 0) == theme.BG


def test_wrap_respects_the_width(c):
    lines = parts.wrap(c, "one two three four five six seven eight nine ten", 120, 12)
    assert len(lines) > 1 and all(c.text_width(ln, 12) <= 120 or " " not in ln for ln in lines)
    assert " ".join(lines) == "one two three four five six seven eight nine ten"
    assert parts.wrap(c, "", 100) == []


def test_paragraph_returns_the_y_below_the_text(c):
    end = parts.paragraph(c, 10, 100, "alpha beta gamma delta epsilon zeta", 60, 12, line_h=20)
    assert end == 100 + 20 * len(parts.wrap(c, "alpha beta gamma delta epsilon zeta", 60, 12))


def test_chip_row_positions_and_progressive_reveal(c):
    boxes = parts.chip_row(c, 20, 100, list("abc"), theme.TOKEN, shown=1.0)
    assert len(boxes) == 3 and boxes[1][0] > boxes[0][0] + boxes[0][1]
    assert px(c, 22, 105) == theme.TOKEN  # first chip visible
    x2 = int(boxes[2][0]) + 3
    assert px(c, x2, 105) == theme.BG  # third chip not revealed yet
    assert parts.row_width(c, list("abc")) == pytest.approx(boxes[2][0] + boxes[2][1] - 20)


def test_chip_row_accepts_one_colour_per_chip(c):
    parts.chip_row(c, 20, 100, list("ab"), [theme.LOSS, theme.VALUE])
    assert px(c, 22, 105) == theme.LOSS


def test_centred_x_centres_the_row(c):
    labels = list("hello")
    left = parts.centred_x(c, labels)
    assert left + parts.row_width(c, labels) / 2 == pytest.approx(400)


def test_panel_title_and_note_draw_text(c):
    parts.panel_title(c, 20, 20, "Title")
    parts.note(c, 20, 60, "note")
    assert c.finish().getbbox() is not None


def test_check_draws_progressively_and_not_at_zero(c):
    parts.check(c, 100, 100, 10, theme.VALUE, 0)
    assert px(c, 100, 100) == theme.BG
    parts.check(c, 100, 100, 10, theme.VALUE, 1)
    assert px(c, 100, 100) != theme.BG
    parts.check(c, 200, 100, 10, theme.VALUE, 0.3)  # half-drawn tick must not fail


def test_bars_reveal_and_value_labels(c):
    parts.bars(c, 20, 100, ["a", "b"], [0.5, 1.0], 200, 20, theme.TOKEN, reveal=1.0)
    img = c.finish()
    assert img.getpixel((20 + 34 + 50, 100 + 10)) == theme.TOKEN  # first bar: half of the track is filled
    assert img.getpixel((20 + 34 + 150, 100 + 10)) == theme.GRID  # ... and the rest is empty
    assert img.getpixel((20 + 34 + 150, 120 + 10)) == theme.TOKEN  # second bar (the maximum) fills the track
    c2 = Canvas(800, 450, 1)
    parts.bars(c2, 20, 100, ["a"], [1.0], 200, 20, reveal=0.0, show_values=False)
    assert c2.finish().getpixel((20 + 34 + 150, 110)) == theme.GRID
    parts.bars(c2, 20, 200, ["a"], [], 200)  # empty data is fine
    parts.bars(c2, 20, 240, ["a"], [0.4], highlight=0)
