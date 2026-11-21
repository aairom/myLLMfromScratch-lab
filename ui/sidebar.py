"""The global model-configuration sidebar (shown on every page)."""
from __future__ import annotations

import streamlit as st

from llm.embeddings import INIT_STRATEGIES
from llm.layers import ACTIVATIONS
from llm.positional import POSITIONAL_KINDS
from training.data import find_input_files
from ui.state import MAX_BPE_MERGES, TOKENIZER_KINDS, ModelSettings, apply_settings, get_device

SOURCE_LABELS = {
    "fresh": "freshly initialised (untrained)",
    "trained": "trained in this session",
    "checkpoint": "loaded from a checkpoint",
    "none": "not built yet",
}


def _index(options, value) -> int:
    """Position of ``value`` in ``options`` (0 if it is not there)."""
    options = list(options)
    return options.index(value) if value in options else 0


def settings_form(current: ModelSettings) -> tuple[bool, ModelSettings]:
    """Render the form; returns ``(submitted, settings_from_widgets)``."""
    with st.form("model_settings"):
        st.subheader("Data and tokenizer")
        use_bundled = st.checkbox("Bundled demo corpus", value=current.use_bundled, key="cfg_bundled")
        n_files = len(find_input_files())
        use_input = st.checkbox(
            f"Files from input/ ({n_files} found)", value=current.use_input, key="cfg_input",
            help="Every .txt or .md file below the input/ folder is added to the training text.",
        )
        tokenizer = st.radio(
            "Tokenizer", TOKENIZER_KINDS, index=_index(TOKENIZER_KINDS, current.tokenizer),
            horizontal=True, key="cfg_tokenizer",
        )
        merges = st.slider(
            "BPE merges (BPE only)", 0, 500, min(current.bpe_merges, 500), key="cfg_merges",
            help=f"More merges = bigger vocabulary and shorter sequences (max {MAX_BPE_MERGES}).",
        )

        st.subheader("Architecture")
        d_model = st.number_input("d_model (embedding width)", 4, 512, current.d_model, step=4, key="cfg_d_model")
        n_heads = st.number_input("Attention heads", 1, 32, current.n_heads, key="cfg_heads")
        n_layers = st.number_input("Transformer layers", 1, 12, current.n_layers, key="cfg_layers")
        max_seq_len = st.number_input("Context window (tokens)", 4, 512, current.max_seq_len, step=4, key="cfg_ctx")
        ffn_mult = st.number_input("FFN width multiplier", 1, 8, current.ffn_mult, key="cfg_ffn")
        activation = st.selectbox(
            "Activation", list(ACTIVATIONS), index=_index(ACTIVATIONS, current.activation), key="cfg_act"
        )
        positional = st.selectbox(
            "Positional encoding", POSITIONAL_KINDS, index=_index(POSITIONAL_KINDS, current.positional), key="cfg_pos"
        )
        embedding_init = st.selectbox(
            "Embedding init", INIT_STRATEGIES, index=_index(INIT_STRATEGIES, current.embedding_init), key="cfg_init"
        )
        dropout = st.slider("Dropout", 0.0, 0.5, float(current.dropout), 0.05, key="cfg_dropout")
        tie = st.checkbox("Tie input and output embeddings", value=current.tie_weights, key="cfg_tie")
        seed = st.number_input("Random seed", 0, 2**31 - 1, current.seed, key="cfg_seed")

        submitted = st.form_submit_button("Apply and re-initialise", type="primary")

    return submitted, ModelSettings(
        tokenizer=tokenizer, bpe_merges=int(merges), use_bundled=use_bundled, use_input=use_input,
        d_model=int(d_model), n_heads=int(n_heads), n_layers=int(n_layers), max_seq_len=int(max_seq_len),
        ffn_mult=int(ffn_mult), activation=activation, positional=positional, embedding_init=embedding_init,
        dropout=float(dropout), tie_weights=tie, seed=int(seed),
    )


def render_sidebar(state) -> None:
    """Draw the sidebar: live model facts, then the settings form."""
    with st.sidebar:
        st.header("Model")
        model = state.get("model")
        if model is not None:
            c1, c2 = st.columns(2)
            c1.metric("Parameters", f"{model.num_parameters():,}")
            c2.metric("Vocabulary", f"{state['tokenizer'].vocab_size:,}")
            st.caption(
                f"Device: {get_device(state)} - weights are {SOURCE_LABELS.get(state['model_source'], '')}"
            )
        if state.get("training"):
            st.warning("Training is running; apply settings after it stops.")

        submitted, settings = settings_form(state["settings"])
        if submitted:
            problems = settings.problems()
            if problems:
                for problem in problems:
                    st.error(problem)
            else:
                error = apply_settings(state, settings)
                if error:
                    st.error(error)
                else:
                    st.success("Model re-initialised. Earlier training was discarded.")
        elif state.get("last_error"):
            st.error(state["last_error"])
