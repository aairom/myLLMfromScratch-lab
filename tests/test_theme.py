"""Unit tests for docsgen/theme.py."""
from docsgen import theme


def test_stylesheet_is_self_contained():
    css = theme.stylesheet()
    assert "@import" not in css and "url(" not in css
    assert "prefers-color-scheme" in css  # light and dark themes


def test_code_css_has_light_and_dark_rules():
    css = theme.code_css()
    assert ".code" in css and "@media (prefers-color-scheme: dark)" in css


def test_stylesheet_contains_the_base_and_extra_parts():
    css = theme.stylesheet()
    assert theme.BASE_CSS in css and theme.EXTRA_CSS in css
