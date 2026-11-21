"""Embeddings: integer ids -> learnable vectors."""
import streamlit as st

from llm.embeddings import INIT_STRATEGIES
from ui import demo
from ui.components import init_page, lesson_tabs, page_header, plot
from viz.embeddings import embedding_heatmap, embedding_scatter, similarity_heatmap

init_page()
page_header("embeddings")
lesson_tabs("embeddings")

tokenizer = st.session_state["tokenizer"]
settings = st.session_state["settings"]

st.divider()
st.subheader("Try it: look up vectors")
col_a, col_b = st.columns(2)
dim = col_a.slider("Embedding width d_model", 2, 128, min(settings.d_model, 128), key="emb_dim")
init = col_b.selectbox("Initialisation", INIT_STRATEGIES, key="emb_init")
scale = st.toggle("Scale by sqrt(d_model)", key="emb_scale", help="As in the original Transformer paper.")
text = st.text_input("Text", demo.DEFAULT_TEXT, key="emb_text")

out = demo.embedding_demo(tokenizer, text, dim, init=init, scale=scale, seed=settings.seed)
if not out["ids"]:
    st.info("Type some text to embed.")
    st.stop()

st.caption("The 'layer' is just row indexing: token id -> row of a (vocabulary x d_model) table.")
plot(embedding_heatmap(out["vectors"], out["tokens"]), "emb_heat")
c1, c2, c3 = st.columns(3)
c1.metric("Table shape", f"{tokenizer.vocab_size} x {dim}")
c2.metric("Table parameters", f"{tokenizer.vocab_size * dim:,}")
c3.metric("Std of values", f"{out['table'].std():.3f}")

st.subheader("Where tokens sit in the table")
left, right = st.columns(2)
with left:
    st.caption("2-D projection (PCA). Untrained tables are random, so no structure yet.")
    labels = [tokenizer.itos[i] for i in range(tokenizer.vocab_size)]
    plot(embedding_scatter(out["table"], labels), "emb_scatter")
with right:
    st.caption("Cosine similarity between the embedded tokens of your text.")
    plot(similarity_heatmap(out["vectors"], out["tokens"]), "emb_sim")
st.info("Train the model in the **Training lab**, then revisit this page: related tokens drift together.")
