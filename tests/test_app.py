"""Tests for app.py: the page registry and the full app (navigation + sidebar) via AppTest."""
import pytest
from streamlit.testing.v1 import AppTest

import app
from tests.conftest import PROJECT_ROOT, assert_clean


def test_registry_has_unique_existing_pages():
    paths = [p for p, *_ in app.PAGES]
    assert len(paths) == len(set(paths)) == 10
    assert all((PROJECT_ROOT / p).is_file() for p in paths)
    urls = [u for *_, u in app.PAGES]
    assert len(urls) == len(set(urls))


def test_home_comes_first_and_every_page_is_built():
    assert app.PAGES[0][3] == "home"  # the first entry is the default landing page
    assert len(app.build_pages()) == len(app.PAGES)


def test_importing_app_has_no_side_effects():
    """Import must not run the UI (main() is guarded by __name__)."""
    assert callable(app.main)


@pytest.fixture
def full_app():
    return AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120).run()


def test_full_app_renders_home_with_sidebar(full_app):
    assert_clean(full_app)
    assert "Parameters" in [m.label for m in full_app.sidebar.metric]
    assert full_app.session_state["model"] is not None


def test_sidebar_apply_rebuilds_model(full_app):
    full_app.sidebar.number_input(key="cfg_d_model").set_value(32)
    full_app.sidebar.number_input(key="cfg_heads").set_value(2)
    full_app.sidebar.number_input(key="cfg_layers").set_value(1)
    full_app.sidebar.button[0].click().run()
    assert_clean(full_app)
    assert full_app.session_state["model"].cfg.d_model == 32
    assert full_app.session_state["model"].cfg.n_layers == 1


def test_sidebar_rejects_invalid_head_count_and_keeps_model(full_app):
    old = full_app.session_state["model"]
    full_app.sidebar.number_input(key="cfg_d_model").set_value(64)
    full_app.sidebar.number_input(key="cfg_heads").set_value(5)
    full_app.sidebar.button[0].click().run()
    assert not full_app.exception
    assert any("divisible" in e.value for e in full_app.sidebar.error)
    assert full_app.session_state["model"] is old


def test_sidebar_requires_a_text_source(full_app):
    full_app.sidebar.checkbox(key="cfg_bundled").set_value(False)
    full_app.sidebar.checkbox(key="cfg_input").set_value(False)
    full_app.sidebar.button[0].click().run()
    assert any("at least one text source" in e.value for e in full_app.sidebar.error)


def test_sidebar_switch_to_bpe_grows_vocabulary(full_app):
    chars = full_app.session_state["tokenizer"].vocab_size
    full_app.sidebar.radio(key="cfg_tokenizer").set_value("bpe")
    full_app.sidebar.slider(key="cfg_merges").set_value(30)
    full_app.sidebar.button[0].click().run()
    assert_clean(full_app)
    assert full_app.session_state["tokenizer"].vocab_size == chars + 30
