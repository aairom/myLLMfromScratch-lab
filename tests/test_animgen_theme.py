"""Unit tests for animgen/theme.py (the docsgen stylesheet is tested in test_theme.py)."""
import pytest

from animgen import theme


def test_clamp_lerp_and_segment():
    assert (theme.clamp01(-1), theme.clamp01(0.4), theme.clamp01(9)) == (0.0, 0.4, 1.0)
    assert theme.lerp(10, 20, 0.5) == 15
    assert (theme.segment(0.0, 0.2, 0.6), theme.segment(1, 0.2, 0.6)) == (0.0, 1.0)
    assert theme.segment(0.4, 0.2, 0.6) == pytest.approx(0.5)
    assert theme.segment(0.5, 0.5, 0.5) == 1.0 and theme.segment(0.4, 0.5, 0.5) == 0.0


def test_easing_curves_start_at_zero_and_end_at_one():
    for f in (theme.ease, theme.ease_out):
        assert f(0) == 0 and f(1) == 1 and 0 < f(0.5) < 1
    assert theme.ease(0.5) == 0.5 and theme.ease_out(0.5) > 0.5


def test_mix_tint_and_alpha():
    assert theme.mix((0, 0, 0), (100, 200, 50), 0.5) == (50, 100, 25)
    assert theme.mix((0, 0, 0), (10, 10, 10), 5) == (10, 10, 10)
    assert theme.tint((0, 0, 0), 0) == (255, 255, 255) and theme.tint((9, 9, 9), 1) == (9, 9, 9)
    assert theme.with_alpha((1, 2, 3), 0.5) == (1, 2, 3, 128)


def test_heat_and_diverging_scales():
    assert theme.heat(0) == (247, 250, 255) and theme.heat(1) == theme.TOKEN and theme.heat(5, 0) == (247, 250, 255)
    assert theme.diverging(0) == (255, 255, 255)
    assert theme.diverging(1) == theme.QUERY and theme.diverging(-1) == theme.TOKEN and theme.diverging(1, 0) == (255, 255, 255)


def test_spread_and_sparkline():
    assert theme.spread(0, 0, 1) == [] and theme.spread(1, 0, 10) == [5] and theme.spread(3, 0, 10) == [0, 5, 10]
    assert theme.sparkline([1, 2, 3, 4], 2) == [1.5, 3.5]
    assert theme.sparkline([], 3) == [] and theme.sparkline([1], 0) == []
    assert len(theme.sparkline([1, 2, 3], 6)) == 6


def test_round_half_and_role_colours():
    assert (theme.round_half(0.5), theme.round_half(1.4), theme.round_half(-0.5)) == (1, 1, 0)
    assert theme.ROLE_COLOURS["query"] == theme.QUERY != theme.ROLE_COLOURS["key"]
    assert theme.ROLE_COLOURS["value"] == theme.VALUE
