"""Unit tests for animgen/canvas.py."""
import pytest
from PIL import Image

from animgen import theme
from animgen.canvas import DESIGN_H, DESIGN_W, Canvas


def blank(w=640, h=360, ss=1) -> Canvas:
    return Canvas(w, h, ss)


def pixel(c: Canvas, x: float, y: float):
    img = c.finish()
    px, py = c.px(x, y)
    return img.getpixel((round(px / c.ss), round(py / c.ss)))


def test_design_space_maps_to_the_pixel_size():
    c = blank(1600, 900, 2)
    assert c.k == pytest.approx(4.0) and c.px(0, 0) == (0, 0)
    assert c.px(DESIGN_W, DESIGN_H) == pytest.approx((3200, 1800))


def test_other_aspect_ratios_are_letterboxed_and_centred():
    c = blank(800, 600)
    assert c.k == 1.0 and c.oy == 75 and c.ox == 0


def test_finish_returns_the_requested_size_with_and_without_supersampling():
    assert blank(640, 360, 2).finish().size == (640, 360)
    assert blank(640, 360, 1).finish().size == (640, 360)


def test_rect_circle_polygon_and_line_draw_pixels():
    c = blank()
    c.rect(10, 10, 60, 40, fill=theme.LOSS)
    c.circle(200, 100, 20, fill=theme.VALUE)
    c.polygon([(300, 100), (340, 100), (320, 140)], theme.ACCENT)
    c.line([(400, 100), (460, 100)], theme.QUERY, 4)
    assert pixel(c, 30, 25) == theme.LOSS and pixel(c, 200, 100) == theme.VALUE
    assert pixel(c, 320, 110) == theme.ACCENT and pixel(c, 430, 100) == theme.QUERY
    assert pixel(c, 700, 300) == theme.BG


def test_rounded_rect_and_one_point_line_are_safe():
    c = blank()
    c.rect(10, 10, 80, 50, fill=theme.TOKEN, radius=8)
    c.line([(5, 5)])  # a single point draws nothing and does not fail
    assert pixel(c, 45, 30) == theme.TOKEN


def test_arrow_has_a_head_and_ignores_zero_length():
    c = blank()
    c.arrow((100, 200), (200, 200), theme.INK, 3, 12)
    c.arrow((50, 50), (50, 50))
    assert pixel(c, 150, 200) == theme.INK and pixel(c, 195, 200) == theme.INK


def test_dashed_line_has_gaps():
    c = blank()
    c.dashed((100, 300), (300, 300), theme.INK, 3, 10)
    seen = {pixel(c, x, 300) == theme.INK for x in range(100, 300, 5)}
    assert seen == {True, False}


def test_text_and_text_width():
    c = blank()
    assert c.text_width("Hello world", 14) > c.text_width("Hello", 14) > 0
    c.text(100, 100, "Hello", 20, fill=theme.INK, anchor="lm")
    assert any(pixel(c, x, 100) != theme.BG for x in range(100, 140))


def test_box_bar_and_token():
    c = blank()
    c.box(20, 100, 120, 40, "Box")
    c.bar(20, 200, 200, 12, 0.5, theme.VALUE)
    w = c.token(20, 250, "ab", theme.TOKEN)
    assert w >= 28 and pixel(c, 21 + w / 2, 252) != theme.BG
    assert pixel(c, 60, 206) == theme.VALUE and pixel(c, 200, 206) == theme.GRID
    c.bar(20, 300, 200, 12, 0.0)  # an empty bar draws only the track


def test_matrix_draws_cells_and_masked_cells_with_values():
    c = blank(1600, 900)
    c.matrix(20, 100, [[0.0, 1.0], [None, 0.5]], (40, 40), fmt=".1f")
    assert pixel(c, 60, 120) == pytest.approx(theme.TOKEN, abs=3) or pixel(c, 60, 120) != theme.BG
    assert pixel(c, 30, 160) == (232, 235, 240)


def test_polyline_reveals_progressively_and_returns_the_last_point():
    c = blank()
    xs, ys = list(range(10)), [i * i for i in range(10)]
    full = c.polyline(xs, ys, (50, 50, 250, 250), (0, 9), (0, 81))
    half = c.polyline(xs, ys, (50, 50, 250, 250), (0, 9), (0, 81), upto=0.5)
    assert full == pytest.approx((250, 50)) and half[0] < full[0]
    assert c.polyline([], [], (0, 0, 1, 1), (0, 1), (0, 1)) is None
    assert c.polyline([1, 1], [2, 2], (0, 0, 10, 10), (1, 1), (2, 2)) is not None  # flat ranges do not divide by zero


def test_axes_and_chrome_render_text_furniture():
    c = blank(1600, 900, 2)
    c.axes((60, 120, 400, 300), "title", "x", "y")
    c.chrome("Sub-task 1", "A title", 1, 4, "A caption")
    img = c.finish()
    assert isinstance(img, Image.Image) and img.size == (1600, 900)

    def ink(box):  # does the region contain any pixel that is not background?
        lo_hi = img.crop(box).getextrema()  # per channel (min, max); a flat background has min == max == BG
        return any(lo != bg or hi != bg for (lo, hi), bg in zip(lo_hi, theme.BG))

    assert ink((0, 0, 700, 130))  # header: kicker, title, progress dots
    assert ink((0, 780, 1600, 900))  # caption bar
    assert ink((100, 230, 820, 620))  # axes frame
    assert not ink((900, 330, 1500, 700))  # empty area stays empty
