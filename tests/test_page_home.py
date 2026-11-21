"""Tests for pages/page_home.py."""
from edu.lessons import all_lessons
from tests.conftest import assert_clean
from ui.state import ModelSettings

SMALL = ModelSettings(d_model=16, n_heads=2, n_layers=1, max_seq_len=16, use_input=False)


def test_home_renders_cleanly(run_page):
    assert_clean(run_page("home", SMALL))


def test_home_lists_every_lesson(run_page):
    at = run_page("home", SMALL)
    text = " ".join(m.value for m in at.markdown)
    assert all(lesson.title in text for lesson in all_lessons())
    assert [m.label for m in at.metric][:2] == ["Parameters", "Layers x heads"]
