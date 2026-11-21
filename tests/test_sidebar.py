"""Tests for ui/sidebar.py: the global model-settings sidebar."""
from streamlit.testing.v1 import AppTest

from tests.conftest import assert_clean


def run(script: str) -> AppTest:
    return AppTest.from_string(script, default_timeout=60).run()


SIDEBAR = (
    "import streamlit as st\n"
    "from ui.state import init_state, ensure_model\n"
    "from ui.sidebar import render_sidebar\n"
    "init_state(st.session_state)\n"
    "ensure_model(st.session_state)\n"
    "render_sidebar(st.session_state)\n"
)


def test_sidebar_shows_model_facts():
    at = run(SIDEBAR)
    assert_clean(at)
    assert {m.label for m in at.sidebar.metric} == {"Parameters", "Vocabulary"}


def test_sidebar_warns_while_training_flag_is_set():
    at = AppTest.from_string(SIDEBAR.replace("render_sidebar(st.session_state)", "st.session_state['training'] = True\nrender_sidebar(st.session_state)"), default_timeout=60).run()
    assert any("Training is running" in w.value for w in at.sidebar.warning)


def test_sidebar_without_model_when_data_is_missing():
    at = run(
        "import streamlit as st\n"
        "from ui.state import ModelSettings, init_state, ensure_model\n"
        "from ui.sidebar import render_sidebar\n"
        "init_state(st.session_state)\n"
        "st.session_state['settings'] = ModelSettings(use_bundled=False, use_input=False)\n"
        "ensure_model(st.session_state)\n"
        "render_sidebar(st.session_state)\n"
    )
    assert not at.exception
    assert any("at least one text source" in e.value for e in at.sidebar.error)
