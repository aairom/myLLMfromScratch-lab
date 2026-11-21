"""Inference: generate text from the current model and inspect each prediction."""
import streamlit as st

from ui import demo
from ui.components import empty_state, init_page, int_slider, lesson_tabs, page_header, plot
from ui.lab import available_checkpoints, load_into_state
from viz.attention import attention_heatmap
from viz.common import empty_figure
import plotly.graph_objects as go

init_page()
page_header("sampling")
lesson_tabs("sampling", extra_sources=("llm.transformer.next_token_distribution", "llm.transformer.GPT.generate"))

state = st.session_state
model, tokenizer = state["model"], state["tokenizer"]

st.divider()
st.subheader("Model to use")
source = state["model_source"]
if source == "fresh":
    st.warning("This model is untrained, so it will produce random characters. Train it in the **Training lab** first.")
else:
    st.success(f"Using a model that was {'trained in this session' if source == 'trained' else 'loaded from a checkpoint'}.")

files = available_checkpoints()
if files:
    with st.expander("Load a saved checkpoint instead"):
        pick = st.selectbox("Checkpoint", files, format_func=lambda p: p.name, key="inf_ckpt")
        if st.button("Load", key="inf_load"):
            error = load_into_state(state, pick)
            if error:
                st.error(error)
            else:
                st.rerun()

st.subheader("Generate")
prompt = st.text_area("Prompt", "The little robot", key="inf_prompt", height=80)
c1, c2, c3, c4 = st.columns(4)
max_new = c1.slider("New tokens", 1, 400, 120, key="inf_max")
temperature = c2.slider("Temperature", 0.0, 2.0, 0.8, 0.05, key="inf_temp",
                        help="0 = always the most likely token; higher = more random.")
top_k = c3.slider("Top-k (0 = off)", 0, 100, 0, key="inf_topk")
top_p = c4.slider("Top-p", 0.05, 1.0, 1.0, 0.05, key="inf_topp")
fixed = st.checkbox("Fixed random seed (reproducible)", value=True, key="inf_fixed")
seed = st.number_input("Seed", 0, 99999, 7, key="inf_seed", disabled=not fixed)

unknown = [ch for ch in prompt if tokenizer.encode(ch) == [0]]
if unknown:
    st.warning(f"These characters are not in the vocabulary and will be treated as unknown: {sorted(set(unknown))}")

if st.button("Generate", type="primary", key="inf_go"):
    out = demo.generate_text(
        model, tokenizer, prompt, max_new, temperature, top_k or None,
        top_p if top_p < 1.0 else None, int(seed) if fixed else None,
    )
    state["inference_result"] = {"out": out, "temperature": temperature, "top_k": top_k or None,
                                 "top_p": top_p if top_p < 1.0 else None}

result = state.get("inference_result")
if not result:
    empty_state("Press **Generate** to continue your prompt.")
    st.stop()

out = result["out"]
st.markdown("**Generated text**")
st.code(out["text"], language=None, wrap_lines=True)
st.caption(f"{len(out['prompt_ids'])} prompt tokens + {len(out['new_ids'])} generated tokens.")

st.subheader("How surprised was the model by each token?")
all_ids = out["prompt_ids"] + out["new_ids"]
rows = demo.step_probabilities(model, tokenizer, all_ids, result["temperature"], result["top_k"], result["top_p"])
if not rows:
    st.info("Need at least two tokens to show per-token probabilities.")
else:
    fig = go.Figure(go.Bar(
        x=[f"{r['position']}·{tokenizer.itos[all_ids[r['position']]]}" for r in rows],
        y=[r["probability"] for r in rows],
        marker_color=["#3b82d4" if r["position"] >= len(out["prompt_ids"]) else "#9aa5b1" for r in rows],
    ))
    fig.update_layout(title="Probability the model assigned to each actual next token (blue = generated)",
                      xaxis=dict(type="category", title="token"), yaxis_title="probability",
                      margin=dict(l=40, r=20, t=50, b=40))
    plot(fig, "inf_probs")

st.subheader("Look inside the last prediction")
tab_next, tab_att = st.tabs(["Next-token candidates", "Attention over the sequence"])
run = demo.model_forward(model, tokenizer, tokenizer.decode(all_ids))
with tab_next:
    rows = demo.top_next_tokens(run["logits"][-1], tokenizer, result["temperature"], result["top_k"], result["top_p"])
    st.dataframe(rows, hide_index=True, width="stretch")
with tab_att:
    layer = int_slider("Layer", 1, model.cfg.n_layers, model.cfg.n_layers, key="inf_layer")
    weights = run["details"]["blocks"][layer - 1]["attention"]["weights"][0].mean(0)
    if len(run["tokens"]) > 80:
        st.info("The sequence is long; showing only the last 80 tokens.")
        plot(attention_heatmap(weights[-80:, -80:], run["tokens"][-80:], "Average attention (last 80 tokens)"), "inf_att")
    else:
        plot(attention_heatmap(weights, run["tokens"], "Average attention over heads"), "inf_att")
