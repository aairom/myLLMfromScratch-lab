"""Multi-head attention: several attention patterns in parallel."""
import streamlit as st

from ui import demo
from ui.components import init_page, int_slider, lesson_tabs, page_header, plot
from viz.attention import attention_heatmap, heads_grid

init_page()
page_header("multihead")
lesson_tabs("multihead", extra_sources=("llm.attention.MultiHeadAttention",))

tokenizer = st.session_state["tokenizer"]
settings = st.session_state["settings"]

st.divider()
st.subheader("Try it: split the width into heads")
text = st.text_input("Text", "the robot read the book", key="mh_text")
c1, c2, c3 = st.columns(3)
d_model = c1.select_slider("d_model", [8, 16, 32, 48, 64, 96, 128], value=32, key="mh_d")
n_heads = c2.number_input("Heads", 1, 16, 4, key="mh_heads")
seed = c3.number_input("Seed", 0, 9999, settings.seed, key="mh_seed")

if d_model % int(n_heads) != 0:
    st.error(
        f"d_model ({d_model}) is not divisible by the number of heads ({int(n_heads)}). "
        f"Each head needs d_model / heads dimensions: try {[h for h in range(1, 17) if d_model % h == 0]}."
    )
    st.stop()

try:
    out = demo.attention_demo(tokenizer, text, d_model, int(n_heads), seed=int(seed))
except ValueError as exc:
    st.info(f"Cannot run attention: {exc}")
    st.stop()

tokens, weights = out["tokens"], out["details"]["weights"]  # weights: (H, T, T)
st.markdown(
    f"**{int(n_heads)} heads x {d_model // int(n_heads)} dimensions each = {d_model}.** "
    "Every head has its own Q, K and V projections, so it can learn a different relationship."
)
plot(heads_grid(weights, tokens), "mh_grid")
head = int_slider("Inspect one head", 0, int(n_heads) - 1, 0, key="mh_head")
plot(attention_heatmap(weights[head], tokens, f"Head {head}"), "mh_one")

st.subheader("Shape bookkeeping")
t = len(tokens)
st.table(
    [
        {"step": "input x", "shape": f"(1, {t}, {d_model})"},
        {"step": "Q, K, V after split", "shape": f"(1, {int(n_heads)}, {t}, {d_model // int(n_heads)})"},
        {"step": "weights", "shape": f"(1, {int(n_heads)}, {t}, {t})"},
        {"step": "heads concatenated", "shape": f"(1, {t}, {d_model})"},
        {"step": "after output projection", "shape": f"(1, {t}, {d_model})"},
    ]
)
st.caption("Untrained heads look similar; after training they specialise (previous token, same word, punctuation...).")
