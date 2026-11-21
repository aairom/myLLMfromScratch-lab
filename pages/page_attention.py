"""Attention: queries, keys and values, scores, softmax, weighted sum."""
import streamlit as st

from ui import demo
from ui.components import init_page, int_slider, lesson_tabs, page_header, plot
from viz.attention import attention_heatmap, attention_received_figure, qkv_heatmap

init_page()
page_header("attention")
lesson_tabs("attention", extra_sources=("llm.attention.scaled_dot_product_attention",))

tokenizer = st.session_state["tokenizer"]
settings = st.session_state["settings"]

st.divider()
st.subheader("Try it: one attention layer on your text")
st.caption("The layer is randomly initialised, so you see the mechanics (masking, row sums), not learned meaning.")
text = st.text_input("Text", "the robot read the book", key="att_text")
c1, c2, c3 = st.columns(3)
causal = c1.toggle("Causal mask (GPT style)", value=True, key="att_causal")
d_model = c2.select_slider("d_model", [8, 16, 32, 64, 128], value=32, key="att_d")
seed = c3.number_input("Seed", 0, 9999, settings.seed, key="att_seed")

try:
    out = demo.attention_demo(tokenizer, text, d_model, n_heads=1, causal=causal, seed=int(seed))
except ValueError as exc:
    st.info(f"Cannot run attention: {exc}")
    st.stop()

tokens, d = out["tokens"], out["details"]
step = st.segmented_control(
    "Pipeline stage", ["1. Q, K, V", "2. Scores", "3. Softmax", "4. Output"], default="1. Q, K, V", key="att_step"
)
if step == "1. Q, K, V":
    st.write("Each token vector is projected three ways: **query** (what I look for), **key** (what I offer) and **value** (what I pass on).")
    q, k, v = st.columns(3)
    with q:
        plot(qkv_heatmap(d["q"][0], tokens, "Q"), "att_q")
    with k:
        plot(qkv_heatmap(d["k"][0], tokens, "K"), "att_k")
    with v:
        plot(qkv_heatmap(d["v"][0], tokens, "V"), "att_v")
elif step == "2. Scores":
    st.write("Scores = Q K^T / sqrt(d_k). Blank cells are masked: a token may not look at the future.")
    plot(attention_heatmap(d["scores"][0], tokens, "Scaled scores (before softmax)", pre_softmax=True), "att_scores")
elif step == "3. Softmax":
    st.write("Softmax turns each row into probabilities that sum to 1.")
    weights = d["weights"][0]
    plot(attention_heatmap(weights, tokens, "Attention weights"), "att_weights")
    row = int_slider("Inspect the row of token", 0, len(tokens) - 1, len(tokens) - 1, key="att_row")
    st.caption(f"Row sum for `{tokens[row]}`: {float(weights[row].sum()):.4f}")
    plot(attention_received_figure(weights, tokens), "att_received")
else:
    st.write("The output of each token is the weighted sum of all value vectors, using its row of weights.")
    plot(qkv_heatmap(d["head_output"][0], tokens, "Attention output"), "att_out")
    st.caption(f"Output shape per head: {tuple(d['head_output'][0].shape)} = (tokens, d_k)")
