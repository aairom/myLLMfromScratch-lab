"""Shared fixtures: run Streamlit pages headlessly with ``AppTest``.

Every page test uses ``run_page`` so the pages are exercised exactly as in the
browser (widgets, session state, charts) without starting a server. The output
folder is redirected to a temporary directory so tests never write to ``output/``.
"""
import logging
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# AppTest warns about missing script-run contexts when pages import helpers; harmless here.
logging.getLogger("streamlit").setLevel(logging.ERROR)


@pytest.fixture(autouse=True)
def isolated_output(tmp_path, monkeypatch):
    """Point OUTPUT_DIR at a temp folder and keep the real input/ folder out of the corpus."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("INPUT_DIR", str(tmp_path / "input"))


@pytest.fixture
def run_page():
    """Return ``run(name, settings=None) -> AppTest`` for ``pages/page_<name>.py``."""

    def _run(name: str, settings=None, timeout: int = 120) -> AppTest:
        at = AppTest.from_file(str(PROJECT_ROOT / "pages" / f"page_{name}.py"), default_timeout=timeout)
        if settings is not None:
            at.session_state["settings"] = settings
        return at.run()

    return _run


def assert_clean(at: AppTest) -> None:
    """A page must render without exceptions or error banners."""
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error, [str(e.value) for e in at.error]
