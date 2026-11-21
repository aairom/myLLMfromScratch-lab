"""LayerNorm and the feed-forward network: the two per-token sub-layers."""
import streamlit as st

from llm.layers import ACTIVATIONS
from ui import demo
from ui.components import init_page, lesson_tabs, page_header, plot
from viz.attention import layernorm_distribution, layernorm_token_stats
from viz.common import heatmap

init_page()
page_header("layernorm")
st.caption("This page covers two lessons: LayerNorm first, then the feed-forward network.")
lesson_tabs("layernorm")

tokenizer = st.session_state["tokenizer"]
settings = st.session_state["settings"]

st.divider()
st.subheader("LayerNorm: re-centre and re-scale every token")
text = st.text_input("Text", "the robot read the book", key="ln_text")
c1, c2 = st.columns(2)
d_model = c1.select_slider("d_model", [8, 16, 32, 64, 128], value=32, key="ln_d")
shift = c2.slider("Artificial offset added to the input", 0.0, 10.0, 3.0, key="ln_shift")

try:
    ln = demo.layernorm_demo(tokenizer, text, d_model, seed=settings.seed, offset=shift)
except ValueError as exc:
    st.info(f"Cannot run LayerNorm: {exc}")
    st.stop()

stages = ln["stages"]
stage = st.segmented_control(
    "Stage", ["mean", "variance", "normalized", "scaled output"], default="normalized", key="ln_stage"
)
if stage == "mean":
    st.write("Step 1: the mean of each token's feature vector (one number per token).")
    st.dataframe([{"token": t, "mean": float(m)} for t, m in zip(ln["tokens"], stages["mean"][:, 0])], hide_index=True)
elif stage == "variance":
    st.write("Step 2: the variance around that mean.")
    st.dataframe([{"token": t, "variance": float(v)} for t, v in zip(ln["tokens"], stages["var"][:, 0])], hide_index=True)
elif stage == "normalized":
    st.write("Step 3: subtract the mean, divide by sqrt(variance + eps): mean 0, std 1 per token.")
    plot(layernorm_distribution(ln["before"], stages["normalized"]), "ln_dist")
    plot(layernorm_token_stats(ln["before"], stages["normalized"], ln["tokens"]), "ln_stats")
else:
    st.write("Step 4: multiply by learnable gamma and add beta. At initialisation gamma = 1 and beta = 0, so it equals step 3.")
    plot(heatmap(stages["output"], y_labels=ln["tokens"], title="LayerNorm output", y_title="token", x_title="feature"), "ln_out")

st.divider()
page_header("ffn")
lesson_tabs("ffn")

st.subheader("Feed-forward network: expand, activate, project back")
f1, f2 = st.columns(2)
mult = f1.slider("Hidden width multiplier", 1, 8, settings.ffn_mult, key="ffn_mult")
act = f2.selectbox("Activation", list(ACTIVATIONS), key="ffn_act")
ffn = demo.ffn_demo(tokenizer, text, d_model, mult, act, seed=settings.seed)
st.markdown(f"**{d_model} -> {mult * d_model} -> {d_model}** ({act}) applied to every token independently.")
plot(heatmap(ffn["hidden"], y_labels=ffn["tokens"], title="Hidden activations", y_title="token", x_title="hidden unit"), "ffn_hidden")
active = float((ffn["hidden"] > 0).float().mean())
st.metric("Hidden units that are positive", f"{active:.0%}")
params = d_model * mult * d_model * 2 + mult * d_model + d_model
st.caption(f"This FFN has {params:,} parameters: usually about two thirds of a transformer block.")
