"""Streamlit building blocks shared by every page.

Keeping them here means each page is only "widgets -> helper -> chart", and the
Explain / Math / Code tabs look and behave the same everywhere.
"""
from __future__ import annotations

import streamlit as st

from edu.lessons import Lesson, get_lesson
from edu.snippets import SnippetError, get_source
from ui.state import ensure_model, init_state


def init_page() -> None:
    """Prepare session state and make sure a model exists; stop the page with a message otherwise."""
    init_state(st.session_state)
    error = ensure_model(st.session_state)
    if error:
        st.error(f"Could not build the model: {error}")
        st.info("Adjust the settings in the sidebar and press **Apply**.")
        st.stop()


def plot(fig, key: str) -> None:
    """Draw a Plotly figure at full width (a key is required to keep reruns stable)."""
    st.plotly_chart(fig, width="stretch", key=key)


def int_slider(label: str, low: int, high: int, value: int, key: str) -> int:
    """Integer slider that tolerates a single option (``st.slider`` rejects ``low == high``)."""
    if high <= low:
        st.caption(f"{label}: {low} (the only option for this model)")
        return low
    return st.slider(label, low, high, min(max(value, low), high), key=key)


def lesson_tabs(key: str, extra_sources: tuple[str, ...] = ()) -> None:
    """Explain / Math / Code tabs for the lesson ``key``."""
    lesson = get_lesson(key)
    explain, math_tab, code = st.tabs(["Explain", "Math", "Code"])
    with explain:
        render_explain(lesson)
    with math_tab:
        render_math(lesson)
    with code:
        render_code(lesson, extra_sources)


def render_explain(lesson: Lesson) -> None:
    st.markdown(lesson.intuition)
    st.markdown("**Step by step**")
    for i, step in enumerate(lesson.steps, start=1):
        st.markdown(f"{i}. {step}")
    if lesson.pitfalls:
        with st.expander("Common pitfalls"):
            for item in lesson.pitfalls:
                st.markdown(f"- {item}")


def render_math(lesson: Lesson) -> None:
    if not lesson.formulas:
        st.caption("This component has no formula of its own.")
    for formula in lesson.formulas:
        st.latex(formula.latex)
        st.caption(formula.meaning)
    if lesson.shapes:
        st.markdown("**Tensor shapes**")
        st.table(
            [{"tensor": t, "shape": s, "meaning": m} for t, s, m in lesson.shapes]
        )


def render_code(lesson: Lesson, extra_sources: tuple[str, ...] = ()) -> None:
    """Show the real source of each snippet; the comments in it are the explanation."""
    st.caption("This is the code that actually runs in this app, read live from the source files.")
    for path in tuple(lesson.snippets) + tuple(extra_sources):
        try:
            source = get_source(path)
        except SnippetError as exc:  # never break a page over a missing snippet
            st.warning(f"Snippet `{path}` is unavailable: {exc}")
            continue
        with st.expander(path, expanded=path == lesson.snippets[0] if lesson.snippets else False):
            st.code(source, language="python")


def page_header(key: str) -> None:
    """Title and one-line summary taken from the lesson."""
    lesson = get_lesson(key)
    st.title(lesson.title)
    st.write(lesson.summary)


def empty_state(message: str, hint: str = "") -> None:
    """A consistent 'nothing to show yet' notice."""
    st.info(message)
    if hint:
        st.caption(hint)
