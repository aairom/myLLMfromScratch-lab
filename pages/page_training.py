"""Training lab: tune hyper-parameters, train live, watch the curves, save checkpoints."""
import math

import streamlit as st

from training.optimizers import OPTIMIZER_NAMES
from training.schedulers import SCHEDULE_NAMES
from training.trainer import TrainConfig
from ui import lab
from ui.components import empty_state, init_page, lesson_tabs, page_header, plot
from viz.training import grad_norm_figure, loss_figure, lr_figure, schedule_figure, throughput_figure

init_page()
st.title("Training lab")
st.write("Pick hyper-parameters, press **Start**, and watch the model learn. Loss should fall from about ln(vocabulary).")
lesson_tabs("optimizer", extra_sources=("training.trainer.train", "training.optimizers.Adam"))

state = st.session_state
model, tokenizer = state["model"], state["tokenizer"]
ctx = model.cfg.max_seq_len

st.divider()
st.subheader("Hyper-parameters")
a, b, c = st.columns(3)
with a:
    steps = st.number_input("Steps", 1, 20000, 300, step=50, key="tr_steps")
    batch = st.number_input("Batch size", 1, 256, 16, key="tr_batch")
    block = st.number_input("Window length", 2, 512, min(32, ctx), key="tr_block", help="Must not exceed the model context.")
with b:
    optimizer = st.selectbox("Optimizer", OPTIMIZER_NAMES, index=OPTIMIZER_NAMES.index("adamw"), key="tr_opt")
    lr = st.number_input("Learning rate", 1e-6, 1.0, 3e-3, format="%.6f", key="tr_lr")
    wd = st.number_input("Weight decay", 0.0, 1.0, 0.01, format="%.4f", key="tr_wd")
with c:
    schedule = st.selectbox("LR schedule", SCHEDULE_NAMES, key="tr_sched")
    warmup = st.number_input("Warm-up steps", 0, 5000, 20, key="tr_warm")
    clip = st.number_input("Gradient clip (0 = off)", 0.0, 100.0, 1.0, key="tr_clip")
d, e, f = st.columns(3)
momentum = d.number_input("SGD momentum", 0.0, 0.999, 0.9, key="tr_mom")
eval_every = e.number_input("Evaluate every N steps", 1, 1000, 25, key="tr_eval")
redraw = f.number_input("Redraw charts every N steps", 1, 100, 5, key="tr_redraw")

try:
    cfg = TrainConfig(
        steps=int(steps), batch_size=int(batch), block_size=int(block), optimizer=optimizer, lr=float(lr),
        weight_decay=float(wd), momentum=float(momentum), schedule=schedule, warmup_steps=int(warmup),
        grad_clip=float(clip), eval_interval=int(eval_every), seed=state["settings"].seed,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

plot(schedule_figure(lab.schedule_preview(cfg), cfg.steps), "tr_schedule")
problems = lab.check_train_config(state, cfg)
for problem in problems:
    st.error(problem)

st.divider()
st.subheader("Run")
start_col, resume_col, stop_col = st.columns(3)
start = start_col.button("Start (from scratch)", type="primary", disabled=bool(problems), key="tr_start")
can_resume = bool(state["history"]) and state.get("optimizer") is not None
resume = resume_col.button("Resume", disabled=bool(problems) or not can_resume, key="tr_resume",
                           help="Continue with the same optimizer state.")
stop = stop_col.button("Stop", key="tr_stop")
if stop:
    state["stop_requested"] = True  # polled by the trainer between steps

status_box = st.empty()
metric_box = st.empty()
loss_box = st.empty()
extra_box = st.empty()
baseline = math.log(tokenizer.vocab_size)


frame = [0]  # counts redraws within this script run so every chart key is unique


def draw(history, live: bool) -> None:
    """Redraw the progress area; ``live`` makes the widget keys unique per frame."""
    frame[0] += 1
    suffix = f"_{frame[0]}" if live else ""
    summary = lab.history_summary(history)
    with metric_box.container():
        m = st.columns(5)
        m[0].metric("Step", summary["step"])
        m[1].metric("Train loss", f"{summary['train_loss']:.3f}")
        m[2].metric("Val loss", f"{summary['val_loss']:.3f}" if summary["val_loss"] is not None else "-")
        m[3].metric("Learning rate", f"{summary['lr']:.2e}")
        m[4].metric("Tokens / s", f"{summary['tokens_per_sec']:,.0f}")
    with loss_box.container():
        plot(loss_figure(history, smooth_window=max(1, len(history) // 20), baseline=baseline), "tr_loss" + suffix)
    with extra_box.container():
        l, m_, r = st.columns(3)
        with l:
            plot(lr_figure(history), "tr_lr_fig" + suffix)
        with m_:
            plot(grad_norm_figure(history), "tr_gn" + suffix)
        with r:
            plot(throughput_figure(history), "tr_tp" + suffix)


if start or resume:
    state["stop_requested"] = False
    try:
        with status_box.status("Training...", expanded=False) as status:
            for metrics in lab.run_training(state, cfg, resume=bool(resume)):
                if metrics["step"] % int(redraw) == 0 or metrics["done"]:
                    status.update(label=f"Training... step {metrics['step']} / {cfg.steps}")
                    draw(state["history"], live=True)
            draw(state["history"], live=True)
            status.update(label=lab.finish_message(state), state="complete")
    except FloatingPointError as exc:
        status_box.error(f"Training diverged: {exc}")
    except ValueError as exc:
        status_box.error(str(exc))
elif state["history"]:
    draw(state["history"], live=False)
else:
    empty_state("No training run yet.", "Press **Start** to begin; curves appear here while it runs.")

st.divider()
st.subheader("Checkpoints")
name = st.text_input("Name", "model", key="tr_ckpt_name")
if st.button("Save current model to output/", key="tr_save", disabled=not state["history"]):
    path = lab.save_current(state, name)
    st.success(f"Saved {path.name}")
files = lab.available_checkpoints()
if files:
    choice = st.selectbox("Saved checkpoints", files, format_func=lambda p: p.name, key="tr_ckpt_pick")
    if st.button("Load selected checkpoint", key="tr_load"):
        error = lab.load_into_state(state, choice)
        if error:
            st.error(error)
        else:
            st.success("Loaded. All pages now use this model.")
            st.rerun()
else:
    st.caption("No checkpoints in output/ yet.")
