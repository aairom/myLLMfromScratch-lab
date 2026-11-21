"""Full transformer: data flow, shapes, parameters and layer-by-layer internals."""
import streamlit as st

from ui import demo
from ui.components import init_page, int_slider, lesson_tabs, page_header, plot
from viz.attention import attention_heatmap, heads_grid
from viz.common import heatmap
from viz.training import block_stages, dataflow_figure, model_stages, parameter_breakdown_figure

init_page()
page_header("model")
lesson_tabs("model", extra_sources=("llm.transformer.TransformerBlock", "llm.transformer.GPT.forward"))

model = st.session_state["model"]
tokenizer = st.session_state["tokenizer"]
cfg = model.cfg

st.divider()
st.subheader("Data flow with tensor shapes")
text = st.text_input("Input text", "the robot read the book", key="tf_text")
try:
    run = demo.model_forward(model, tokenizer, text)
except ValueError as exc:
    st.info(f"Cannot run the model: {exc}")
    st.stop()
t = len(run["ids"])
left, right = st.columns(2)
with left:
    plot(dataflow_figure(model_stages(cfg, 1, t), "Whole model"), "tf_flow")
with right:
    plot(dataflow_figure(block_stages(cfg, 1, t), "Inside one block"), "tf_block")
if len(tokenizer.encode(text)) > cfg.max_seq_len:
    st.warning(f"Text was cut to the context window of {cfg.max_seq_len} tokens.")

st.subheader("Parameters")
summary = demo.parameter_summary(model)
c1, c2 = st.columns(2)
c1.metric("Total parameters", f"{summary['total']:,}")
c2.metric("Approx. size (float32)", f"{summary['megabytes']:.2f} MB")
plot(parameter_breakdown_figure(summary["breakdown"]), "tf_params")

st.subheader("Follow the tokens through the layers")
details = run["details"]
layer = int_slider("Layer", 1, cfg.n_layers, 1, key="tf_layer")
block = details["blocks"][layer - 1]
tabs = st.tabs(["Residual stream", "Attention", "Next-token prediction"])
with tabs[0]:
    stage = st.selectbox(
        "Tensor", ["input", "ln1", "attn_out", "after_attn", "ln2", "ffn_out", "output"], index=6, key="tf_stage"
    )
    plot(heatmap(block[stage][0], y_labels=run["tokens"], title=f"Layer {layer}: {stage}", y_title="token", x_title="feature"), "tf_stream")
    delta = float((block["output"] - block["input"]).norm() / block["input"].norm())
    st.caption(f"The block changed the residual stream by {delta:.0%} of its size: it adds to the stream rather than replacing it.")
with tabs[1]:
    weights = block["attention"]["weights"][0]
    plot(heads_grid(weights, run["tokens"]), "tf_heads")
    plot(attention_heatmap(weights.mean(0), run["tokens"], "Average over heads"), "tf_avg")
with tabs[2]:
    position = int_slider("Predict what follows token number", 1, t, t, key="tf_pos")
    st.caption(f"After reading `{''.join(run['tokens'][:position])}`")
    c1, c2, c3 = st.columns(3)
    temperature = c1.slider("Temperature", 0.0, 2.0, 1.0, 0.1, key="tf_temp")
    top_k = c2.slider("Top-k (0 = off)", 0, 50, 0, key="tf_topk")
    top_p = c3.slider("Top-p", 0.05, 1.0, 1.0, 0.05, key="tf_topp")
    rows = demo.top_next_tokens(run["logits"][position - 1], tokenizer, temperature, top_k or None, top_p if top_p < 1 else None)
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("An untrained model is close to uniform: every probability is near 1 / vocabulary size.")
