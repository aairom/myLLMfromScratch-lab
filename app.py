"""Streamlit entry point: page configuration, shared state, sidebar and navigation.

Run with ``streamlit run app.py`` (or ``scripts/start.sh``). Each page lives in
``pages/`` as a small script; the logic they use is in ``ui/``, ``llm/``,
``training/`` and ``viz/``.
"""
from __future__ import annotations

import streamlit as st

from ui.sidebar import render_sidebar
from ui.state import ensure_model, init_state

# (file, title, icon, url path) in teaching order.
PAGES = (
    ("pages/page_home.py", "Home", ":material/home:", "home"),
    ("pages/page_tokenization.py", "1. Tokenization", ":material/text_fields:", "tokenization"),
    ("pages/page_embeddings.py", "2. Embeddings", ":material/scatter_plot:", "embeddings"),
    ("pages/page_positional.py", "3. Positional encoding", ":material/waves:", "positional"),
    ("pages/page_attention.py", "4. Attention", ":material/visibility:", "attention"),
    ("pages/page_multihead.py", "5. Multi-head attention", ":material/grid_view:", "multihead"),
    ("pages/page_norm_ffn.py", "6. LayerNorm and FFN", ":material/tune:", "norm-ffn"),
    ("pages/page_transformer.py", "7. Full transformer", ":material/account_tree:", "transformer"),
    ("pages/page_training.py", "8. Training lab", ":material/model_training:", "training"),
    ("pages/page_inference.py", "9. Inference", ":material/chat:", "inference"),
)


def build_pages() -> list:
    """The ``st.Page`` objects, with Home as the default landing page."""
    return [
        st.Page(path, title=title, icon=icon, url_path=url, default=(i == 0))
        for i, (path, title, icon, url) in enumerate(PAGES)
    ]


def main() -> None:
    st.set_page_config(page_title="LLM from scratch", page_icon=":material/neurology:", layout="wide")
    init_state(st.session_state)
    ensure_model(st.session_state)  # an error is shown in the sidebar and on each page
    render_sidebar(st.session_state)
    st.navigation(build_pages()).run()


if __name__ == "__main__":  # Streamlit executes the entry script as __main__
    main()
