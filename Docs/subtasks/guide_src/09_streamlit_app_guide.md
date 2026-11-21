# Implementation guide: the Streamlit application

## Role in the project

`app.py` is the entry point that `scripts/start.sh` launches. It prepares the shared session state, draws the sidebar and registers the ten pages. Everything else the learner sees is a page script that calls helpers from `ui/`, figure builders from `viz/` and lessons from `edu/`.

```flow Request path for one rerun
 browser event (widget changed)
        |
        v
 Streamlit re-executes app.py:main()
        |-- st.set_page_config(layout="wide")
        |-- init_state(st.session_state)           fill missing keys, never overwrite
        |-- ensure_model(st.session_state)         build default model on first use
        |-- render_sidebar(st.session_state)       metrics, settings form, Apply
        '-- st.navigation(build_pages()).run()     executes the selected pages/page_*.py
                     |
                     v
              page script:  init_page() -> page_header -> lesson_tabs -> widgets -> ui.demo / ui.lab -> viz -> plot(fig, key)
```

## Module map

| File | Lines | Contents |
|---|---|---|
| `app.py` | 46 | `PAGES`, `build_pages`, `main` |
| `ui/state.py` | 235 | `ModelSettings`, `default_settings`, `init_state`, `make_tokenizer`, `build_model`, `apply_settings`, `ensure_model`, `settings_from_model`, `adopt_checkpoint`, `get_device` |
| `ui/sidebar.py` | 102 | `settings_form`, `render_sidebar`, `SOURCE_LABELS` |
| `ui/components.py` | 98 | shared widgets and the lesson tabs (see sub-task 8) |
| `ui/lab.py` | 147 | `prepare_data`, `check_train_config`, `schedule_preview`, `run_training`, `finish_message`, `save_current`, `available_checkpoints`, `load_into_state`, `history_summary` |
| `ui/demo.py` | 255 | the non-Streamlit demonstrations used by the learning pages |
| `pages/` | | ten scripts, `page_home.py` 37 lines to `page_training.py` 138 lines |
| `tests/` | | `conftest.py`, `test_app.py` 8, `test_state.py` 25, `test_sidebar.py` 3, `test_lab.py` 19, `test_demo.py` 32, ten `test_page_*.py` |

## The page registry

{{source:app.build_pages}}

{{file:app.py|python|14-26}}

| # | File | Title | URL path |
|---|---|---|---|
| 0 | `pages/page_home.py` | Home (default) | `home` |
| 1 | `pages/page_tokenization.py` | 1. Tokenization | `tokenization` |
| 2 | `pages/page_embeddings.py` | 2. Embeddings | `embeddings` |
| 3 | `pages/page_positional.py` | 3. Positional encoding | `positional` |
| 4 | `pages/page_attention.py` | 4. Attention | `attention` |
| 5 | `pages/page_multihead.py` | 5. Multi-head attention | `multihead` |
| 6 | `pages/page_norm_ffn.py` | 6. LayerNorm and FFN | `norm-ffn` |
| 7 | `pages/page_transformer.py` | 7. Full transformer | `transformer` |
| 8 | `pages/page_training.py` | 8. Training lab | `training` |
| 9 | `pages/page_inference.py` | 9. Inference | `inference` |

The page titles number the model-building stages 1 to 9, which is a different numbering from the ten sub-tasks of the project documentation.

{{source:app.main}}

Importing `app` has no side effects: `main()` runs only under `if __name__ == "__main__"`, which is how Streamlit executes the entry script. A test relies on this to import the page table.

## Settings

{{source:ui.state.ModelSettings}}

| Field | Default | Valid values | Becomes |
|---|---|---|---|
| `tokenizer` | `char` | `char`, `bpe` | tokenizer kind |
| `bpe_merges` | 100 | 0 to `MAX_BPE_MERGES` (2000); the slider stops at 500 | BPE merges |
| `use_bundled` | `True` | bool | include `data/sample_corpus.txt` |
| `use_input` | `True` | bool | include files from `INPUT_DIR` |
| `d_model` | 64 | positive, divisible by `n_heads` | `GPTConfig.d_model` |
| `n_heads` | 4 | positive | `GPTConfig.n_heads` |
| `n_layers` | 2 | positive | `GPTConfig.n_layers` |
| `max_seq_len` | 64 | positive | context window |
| `ffn_mult` | 4 | positive | FFN hidden multiplier |
| `activation` | `gelu` | `gelu`, `relu` | FFN activation |
| `positional` | `sinusoidal` | `sinusoidal`, `learned` | position encoding |
| `embedding_init` | `normal` | `normal`, `uniform`, `xavier` | token table init |
| `dropout` | 0.0 | [0, 1) | dropout probability |
| `tie_weights` | `True` | bool | share embedding and head |
| `seed` | `DEFAULT_SEED` (42) | any int | seed of `torch.manual_seed` |

`problems()` returns a list of readable messages, so several errors are reported at once. `gpt_config(vocab_size)` builds the `GPTConfig`; the vocabulary size is known only after the tokenizer exists.

## Session state

{{source:ui.state.init_state}}

```flow Keys of st.session_state
settings        ModelSettings (frozen)
model           GPT | None
tokenizer       CharTokenizer | BPETokenizer | None
history         list[dict]                  metrics of the trainer, one per step
train_cfg       TrainConfig | None
optimizer       torch.optim.Optimizer | None   alive only inside the session
model_source    "none" | "fresh" | "trained" | "checkpoint"
stop_requested  bool                        polled by the trainer between steps
training        bool                        True while a run is in progress
last_error      str | None
```

Defaults are `copy.deepcopy`'d so two sessions never share the history list.

```flow Transitions of model_source
none ──ensure_model──▶ fresh
fresh | trained | checkpoint ──Apply (apply_settings)──▶ fresh      history, optimizer, train_cfg reset
fresh | checkpoint ──run_training yields ≥ 1 step──▶ trained
any ──load_into_state (adopt_checkpoint)──▶ checkpoint              optimizer = None
```

## Building the model

{{source:ui.state.build_model}}

{{source:ui.state.apply_settings}}

`apply_settings` replaces `settings`, `model`, `tokenizer`, `history`, `train_cfg`, `optimizer`, `model_source`, `stop_requested` and `last_error` in a single `state.update(...)`. On failure it only stores `last_error` and returns the message; the previous model stays in place.

{{source:ui.state.adopt_checkpoint}}

`_cached_tokenizer` is an `lru_cache(maxsize=8)` keyed by `(kind, corpus text, merges)`, so a rerun that does not change the corpus or the merge count does not retrain the BPE tokenizer. The seed is applied with `torch.manual_seed(settings.seed)` immediately before the `GPT` is created, so equal settings give equal initial weights.

## The sidebar

{{source:ui.sidebar.render_sidebar}}

The settings live in an `st.form` named `model_settings`. Widget values are read only when the user presses **Apply and re-initialise**, which avoids rebuilding the model on every change.

| Group | Widgets (keys) |
|---|---|
| Data and tokenizer | `cfg_bundled`, `cfg_input` (with the number of files found), `cfg_tokenizer`, `cfg_merges` |
| Architecture | `cfg_d_model` 4 to 512, `cfg_heads` 1 to 32, `cfg_layers` 1 to 12, `cfg_ctx` 4 to 512, `cfg_ffn` 1 to 8, `cfg_act`, `cfg_pos`, `cfg_init`, `cfg_dropout` 0 to 0.5, `cfg_tie`, `cfg_seed` |

While `state["training"]` is true the sidebar shows a warning that settings should be applied only after the run ends.

## The training page and the lab helpers

{{source:ui.lab.run_training}}

{{source:ui.lab.prepare_data}}

```flow pages/page_training.py control flow
widgets -> TrainConfig(...)  ValueError -> st.error + st.stop
        |
        v
check_train_config   window > context -> error, Start and Resume disabled
        |
 Start / Resume pressed?
   yes: stop_requested = False
        for metrics in lab.run_training(state, cfg, resume):
            every N steps (default 5) or on the last step: status.update + draw(history, live=True)
        FloatingPointError -> "Training diverged"     ValueError -> message
   no : history present -> draw(history, live=False)     else -> empty_state
        |
        v
checkpoints: Save (needs history), pick and Load (rerun on success)
```

The Resume button is enabled only when `state["history"]` is non-empty **and** `state["optimizer"]` is not `None`. A checkpoint load sets the optimiser to `None`, so Resume is unavailable after loading a file.

### Unique chart keys

```python
frame = [0]                       # counts redraws within one script run
suffix = f"_{frame[0]}" if live else ""
plot(loss_figure(...), "tr_loss" + suffix)
```

A key may not repeat within a run. Redrawing the same chart many times while streaming therefore needs a new key per frame.

## The demonstration helpers

`ui/demo.py` is Streamlit-free and returns plain data to the pages:

| Function | Used by | Returns |
|---|---|---|
| `tokenize_text`, `replay_merges`, `compression_counts` | tokenization | tokens, ids, merge replay, token counts per merge budget |
| `embedding_demo`, `positional_demo`, `embed_with_position` | embeddings, positional | vectors and tables from fresh layers |
| `attention_demo` | attention, multihead | internals of one untrained attention layer |
| `layernorm_demo`, `ffn_demo` | norm and FFN | before and after tensors |
| `model_forward`, `top_next_tokens` | transformer, inference | logits, details and the most likely next tokens |
| `generate_text`, `step_probabilities` | inference | sampled text and per-position probabilities |
| `parameter_summary` | transformer | parameter breakdown |

`model_forward` switches to `eval()` and `no_grad`, and restores the previous mode in a `finally` block. `generate_text` starts from token id 1 when the prompt encodes to nothing.

## Environment variables

| Variable | Read by | Meaning |
|---|---|---|
| `APP_PORT` | `config.load_settings`, `scripts/start.sh` | port of the server (8601 by default; 5000 is rejected) |
| `APP_HOST` | `scripts/start.sh` | bind address, `localhost` by default |
| `APP_START_CMD` | `scripts/start.sh` | the command that is run to start Streamlit |
| `DEVICE` | `config.resolve_device` | `auto`, `cpu`, `mps` or `cuda`; an unavailable choice falls back to CPU |
| `DEFAULT_SEED` | `ui.state.default_settings` | initial seed in the sidebar |
| `INPUT_DIR`, `OUTPUT_DIR` | data and checkpoint modules | folders for extra text and for checkpoints |

## Runtime behaviour

- Each rerun executes `app.py` and then the selected page, so state that must persist is in `st.session_state` only.
- The model is built once per session and rebuilt only after **Apply**.
- The device comes from `config.resolve_device`: CUDA, then MPS, then CPU for `auto`.
- A failure to build the model does not crash a page: `init_page` shows the error and an instruction, then calls `st.stop()`.
- A training run occupies the session until it ends or Stop is pressed.

## Tests

`conftest.py` redirects `OUTPUT_DIR` and `INPUT_DIR` to temporary folders for every test, so tests never write to `output/` nor read real input files. The `run_page` fixture runs `pages/page_<name>.py` with `AppTest`, optionally with prepared settings. `assert_clean` fails on any exception or error banner. The state and lab tests use plain dictionaries. `test_app.py` runs the complete app, clicks **Apply** after changing widgets and checks both the success path and that invalid settings keep the old model. `test_page_training.py` covers a run that fills the history, loss reduction, resume, a window longer than the context, a diverging run that shows a message instead of crashing, every optimiser, the constant schedule, too little text, and checkpoint save, load and corruption.

## Common pitfalls

- **Storing a value in a local variable and expecting it on the next rerun.** Use `st.session_state`.
- **Two charts with one key.** Streamlit raises a duplicate-element error.
- **Changing settings during a run.** The sidebar warns; apply them after the run ends.
- **Port 5000.** Reserved by AirDrop on macOS; the configuration rejects it.
- **Expecting Resume after loading a checkpoint.** The optimiser is not restored from a file.
