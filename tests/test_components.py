"""Tests for ui/components.py through tiny AppTest scripts."""
import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import assert_clean


def run(script: str) -> AppTest:
    return AppTest.from_string(script, default_timeout=60).run()


# ----- components -------------------------------------------------------------- #
def test_int_slider_returns_only_option_without_slider():
    at = run(
        "import streamlit as st\n"
        "from ui.components import int_slider\n"
        "st.session_state['v'] = int_slider('Layer', 1, 1, 1, key='k')\n"
    )
    assert not at.exception
    assert at.session_state["v"] == 1
    assert len(at.slider) == 0 and "only option" in at.caption[0].value


def test_int_slider_draws_slider_for_a_range_and_clamps_value():
    at = run(
        "import streamlit as st\n"
        "from ui.components import int_slider\n"
        "st.session_state['v'] = int_slider('Layer', 1, 4, 99, key='k')\n"
    )
    assert not at.exception and at.session_state["v"] == 4 and len(at.slider) == 1


def test_lesson_tabs_show_explain_math_and_code_for_every_lesson():
    from edu.lessons import all_lessons

    for lesson in all_lessons():
        at = run(
            "import streamlit as st\n"
            "from ui.components import init_page, lesson_tabs, page_header\n"
            f"page_header('{lesson.key}')\nlesson_tabs('{lesson.key}')\n"
        )
        assert_clean(at)
        assert [t.label for t in at.tabs] == ["Explain", "Math", "Code"]
        assert len(at.code) >= 1  # the live source appears


def test_render_code_survives_a_missing_snippet():
    at = run(
        "import streamlit as st\n"
        "from ui.components import render_code\n"
        "from edu.lessons import get_lesson\n"
        "render_code(get_lesson('attention'), extra_sources=('llm.attention.DoesNotExist',))\n"
    )
    assert not at.exception
    assert any("unavailable" in w.value for w in at.warning)


def test_empty_state_shows_message_and_hint():
    at = run("from ui.components import empty_state\nempty_state('Nothing yet', 'Press start')\n")
    assert at.info[0].value == "Nothing yet" and at.caption[0].value == "Press start"


def test_init_page_stops_with_clear_message_when_model_cannot_be_built():
    at = run(
        "import streamlit as st\n"
        "from ui.state import ModelSettings\n"
        "from ui.components import init_page\n"
        "st.session_state['settings'] = ModelSettings(d_model=30, n_heads=4)\n"
        "init_page()\n"
        "st.write('should not be reached')\n"
    )
    assert not at.exception
    assert any("divisible" in e.value for e in at.error)
    assert not any("should not be reached" in m.value for m in at.markdown)


def test_plot_draws_a_figure():
    at = run(
        "import plotly.graph_objects as go\n"
        "from ui.components import plot\n"
        "plot(go.Figure(go.Scatter(x=[1, 2], y=[1, 2])), 'k')\n"
    )
    assert not at.exception
