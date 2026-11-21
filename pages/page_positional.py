"""Positional encoding: injecting word order."""
import streamlit as st

from llm.positional import POSITIONAL_KINDS
from ui import demo
from ui.components import init_page, lesson_tabs, page_header, plot
from viz.embeddings import cosine_similarity_matrix, positional_curves, positional_heatmap
from viz.common import heatmap

init_page()
page_header("positional")
lesson_tabs("positional", extra_sources=("llm.positional.sinusoidal_table",))

settings = st.session_state["settings"]

st.divider()
st.subheader("Try it: inspect the position matrix")
c1, c2, c3 = st.columns(3)
kind = c1.selectbox("Kind", POSITIONAL_KINDS, index=POSITIONAL_KINDS.index(settings.positional), key="pos_kind")
dim = c2.slider("d_model", 4, 128, min(settings.d_model, 128), step=2, key="pos_dim")
length = c3.slider("Positions", 4, 128, min(settings.max_seq_len, 64), key="pos_len")

table = demo.positional_demo(kind, dim, length, seed=settings.seed)
plot(positional_heatmap(table, f"{kind} positional encoding"), "pos_heat")
if kind == "learned":
    st.info("A learned table starts as small random numbers and only becomes meaningful during training.")
else:
    st.caption("Low dimensions (left) oscillate quickly, high dimensions (right) slowly: like clock hands.")
    dims = st.multiselect("Dimensions to plot", list(range(dim)), default=[d for d in (0, 1, 4, 5) if d < dim], key="pos_dims")
    plot(positional_curves(table, dims or None), "pos_curves")

st.subheader("Do nearby positions look alike?")
sim = cosine_similarity_matrix(table)
plot(
    heatmap(sim, title="Cosine similarity between positions", colorscale="Viridis", zmid=None,
            x_title="position", y_title="position"),
    "pos_sim",
)
st.caption("The bright diagonal band shows neighbours are similar and far-apart positions differ.")
