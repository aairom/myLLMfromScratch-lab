"""Home: what this lab is, the learning path and the current model at a glance."""
import streamlit as st

from edu.lessons import all_lessons
from ui.components import init_page
from viz.training import dataflow_figure, model_stages
from ui.components import plot

init_page()

st.title("Build a language model from scratch")
st.write(
    "This lab walks through every part of a GPT-style, decoder-only language model: how text becomes "
    "numbers, how the transformer mixes information, and how training turns random weights into "
    "something that writes. Everything is written by hand in PyTorch and runs locally."
)

st.subheader("Learning path")
st.markdown(
    "1. **Tokenization** to **Full transformer**: follow the pages in order, or jump to any of them.\n"
    "2. **Training lab**: train the model and watch the loss fall in real time.\n"
    "3. **Inference**: generate text and see how sampling settings change it.\n\n"
    "Change the model in the sidebar at any time; every page updates to match."
)

for number, lesson in enumerate(all_lessons(), start=1):
    st.markdown(f"**{number}. {lesson.title}** - {lesson.summary}")

model = st.session_state["model"]
st.subheader("Your current model")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Parameters", f"{model.num_parameters():,}")
c2.metric("Layers x heads", f"{model.cfg.n_layers} x {model.cfg.n_heads}")
c3.metric("d_model", model.cfg.d_model)
c4.metric("Context", model.cfg.max_seq_len)
plot(dataflow_figure(model_stages(model.cfg, 1), title="One forward pass"), "home_flow")
st.caption("Open **Training lab** to train it, or **Inference** to see what an untrained model produces.")
