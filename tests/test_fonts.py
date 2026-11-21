"""Unit tests for animgen/fonts.py."""
import pytest

from animgen import fonts


@pytest.fixture(autouse=True)
def reset_fonts():
    fonts.configure()
    yield
    fonts.configure()


def test_visible_makes_whitespace_readable():
    assert fonts.visible(" ") == "_" and fonts.visible("\n") == "/n" and fonts.visible("a") == "a"
    assert fonts.visible_token(" a\t") == "_a/t"


def test_load_returns_a_font_that_measures_text():
    for kind in (fonts.SANS, fonts.BOLD, fonts.MONO):
        assert fonts.load(kind, 16).getlength("Hello") > 0


def test_safe_text_keeps_ascii_and_replaces_missing_glyphs():
    assert fonts.safe_text(fonts.SANS, "abc 123") == "abc 123"
    assert fonts.safe_text(fonts.SANS, "a\ue000b") == "a?b"  # a private-use character no font draws


def test_explicit_missing_font_file_is_an_error(tmp_path):
    fonts.configure(sans=str(tmp_path / "nope.ttf"))
    with pytest.raises(FileNotFoundError):
        fonts.find_font_file(fonts.SANS)


def test_explicit_font_file_is_used(tmp_path):
    found = fonts.find_font_file(fonts.SANS)
    if found is None:
        pytest.skip("no system TrueType font")
    copy = tmp_path / "f.ttf"
    copy.write_bytes(found.read_bytes())
    fonts.configure(sans=str(copy))
    assert fonts.find_font_file(fonts.SANS) == copy
    assert fonts.load(fonts.SANS, 14).getlength("x") > 0


def test_unreadable_font_falls_back_to_the_built_in_one(tmp_path):
    bad = tmp_path / "bad.ttf"
    bad.write_text("not a font")
    fonts.configure(mono=str(bad))
    assert fonts.load(fonts.MONO, 12).getlength("x") > 0
