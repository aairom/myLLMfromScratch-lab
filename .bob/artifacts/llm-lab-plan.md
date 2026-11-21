# LLM From Scratch Lab - Implementation Plan

## Top-Level Overview

**Goal:** A fully local, educational Streamlit application that walks a learner through building a GPT-style decoder-only Large Language Model from scratch, with interactive visualizations, code and math alongside each component, a live training lab and an inference page.

**Approach (confirmed with user):**
- Python + Streamlit (UI), PyTorch (tensors and autograd only), Plotly (charts).
- Every model component is hand-written from `torch.nn.Module` / raw tensors. `nn.MultiheadAttention`, `nn.TransformerEncoder/Decoder`, `nn.LayerNorm` are NOT used (they may be used only inside tests as reference implementations).
- Optimizers (Adam, SGD with momentum) are hand-written and verified against `torch.optim`; `torch.optim` is offered as an extra option.
- Tokenizers: character-level and a small hand-written BPE.
- Dataset: small bundled public-domain text, plus optional `.txt` files read recursively from `input/`.
- Model size: roughly 0.1M-1M parameters so training runs on a laptop CPU or Apple MPS.
- Checkpoints and metrics saved to `output/` with `YYYYmmdd_HHMMSS_<name>` names.

**Non-goals:** cloud services, multi-GPU/distributed training, large-scale pretraining, fine-tuning/RLHF, containerization (not requested).

**Project rules that apply (from `AGENTS.md` and `.bob/skills/`):**
- `uv` + `.venv` (skill `python-env`); `requirements.txt` is the dependency source.
- Config (ports, device, paths, defaults) from `.env`, documented in `.env.example`; no hard-coded URLs/keys (skill `secure-config`). Never use port 5000 (macOS AirDrop).
- Source directly under project root; docs in `Docs/` (only `README.md` at root); scripts in `scripts/` and executable (skill `project-lifecycle-scripts`).
- `input/` and `output/` with `.gitkeep`, content git-ignored (skill `io-folders`); `.gitignore` via skill `git-hygiene`.
- Every module unit tested; completion requires `test-gate` exit code 0 (skill `test-gate`).
- README must contain architecture and workflow Mermaid diagrams and the License type (skill `project-docs`). `project-audit` run at the end.

## Target Layout

```
.env.example
.gitignore
requirements.txt
README.md
app.py                      # Streamlit entry: navigation + home
pages/                      # one file per learning page (numbered)
llm/                        # core model package (pure PyTorch, no Streamlit imports)
  tokenizer.py              # CharTokenizer, BPETokenizer
  embeddings.py             # TokenEmbedding, init strategies
  positional.py             # sinusoidal + learned positional encoding
  attention.py              # scaled dot-product, MultiHeadAttention (returns weights)
  layers.py                 # LayerNorm, FeedForward (GELU), residual helpers
  transformer.py            # TransformerBlock, GPT model, generate()
training/
  data.py                   # corpus loading, split, batching
  optimizers.py             # Adam, SGD (momentum)
  schedulers.py             # warmup + cosine decay
  trainer.py                # training loop as generator yielding metrics
  checkpoint.py             # save/load model + optimizer + config + vocab
viz/                        # Plotly figure builders (no model logic)
edu/                        # explanations: markdown text, LaTeX formulas, code snippet loader
data/                       # bundled sample corpus
tests/                      # unit tests mirroring the packages
scripts/                    # start.sh, stop.sh, cleanup.sh
Docs/                       # Quickstart.md, Architecture.md
input/ output/              # .gitkeep
```

Separation of concerns: `llm/` and `training/` know nothing about Streamlit; `viz/` only turns tensors/metrics into figures; `pages/` only wires widgets to those layers. Code snippets shown in the UI are loaded from the real source files via `inspect`/file reading so the displayed code can never drift from what runs.

## Sub-Tasks

Process one at a time; update Status after each is finished and verified.

---

### 1. Project scaffolding and environment
**Intent:** Establish the project skeleton and tooling that follows the workspace rules before any feature code.
**Expected Outcomes:** `.venv` created with `uv`; dependencies installed; `.gitignore`, `.env.example`, `input/` and `output/` exist; empty package structure importable.
**Todo List:**
1. Create `requirements.txt` (torch, streamlit, plotly, numpy, pytest, python-dotenv).
2. Run `python-env` skill script to build `.venv`.
3. Run `git-hygiene` skill to create `.gitignore`.
4. Create `.env.example` (e.g. `APP_PORT`, `DEVICE` auto/cpu/mps/cuda, `DEFAULT_SEED`, `OUTPUT_DIR`, `INPUT_DIR`) and a small config loader module that reads `.env`.
5. Create `input/.gitkeep`, `output/.gitkeep`, package folders with `__init__.py`, `tests/` folder.
**Relevant Context:** `.bob/skills/python-env/SKILL.md`, `.bob/skills/git-hygiene/SKILL.md`, `.bob/skills/io-folders/SKILL.md`, `.bob/skills/secure-config/SKILL.md`, `.bob/rules/macos.md` (avoid port 5000).
**Status:** [x] done (also added `config.py`, `pytest.ini`; `.venv` uses Python 3.13, torch 2.15, MPS available; 5 tests pass)

---

### 2. Tokenization module
**Intent:** Teach how text becomes integer ids and how vocabularies are built.
**Expected Outcomes:** `CharTokenizer` and `BPETokenizer` with encode/decode round-trip, vocab building, special tokens; BPE exposes the ordered merge list and per-step merge trace for visualization.
**Todo List:**
1. Implement `CharTokenizer` (build vocab from corpus, encode, decode, unknown handling).
2. Implement `BPETokenizer` (train N merges, record each merge step with pair counts, encode using merges, decode).
3. Serialize/deserialize tokenizer state (needed by checkpoints).
4. Unit tests: round-trip, unknown chars, merge ordering, empty input, determinism.
**Relevant Context:** `llm/tokenizer.py`, `tests/test_tokenizer.py`.
**Status:** [x] done (`llm/tokenizer.py`, 19 tests pass). Notes for later sub-tasks: both tokenizers expose `tokens()`, `encode()`, `decode()`, `vocab_size`, `to_dict()`; factory helpers `build_tokenizer(kind, text, num_merges)` and `tokenizer_from_dict(state)`; `BPETokenizer.trace` holds `MergeStep` records (step, pair, count, new_token, top_pairs) for the UI; id 0 is always `<unk>`.

---

### 3. Embeddings and positional encoding
**Intent:** Teach how ids become vectors and how order information is injected.
**Expected Outcomes:** `TokenEmbedding` with selectable init (normal, uniform, Xavier) and configurable `d_model`; sinusoidal and learned positional encodings, both returning inspectable matrices.
**Todo List:**
1. Implement `TokenEmbedding` as an explicit lookup table parameter (no `nn.Embedding`), with init strategy option and sqrt(d_model) scaling option.
2. Implement sinusoidal positional encoding (formula `PE(pos,2i)=sin(pos/10000^(2i/d))`, cosine for odd) and learned positional embedding.
3. Unit tests: shapes, sinusoidal values vs closed-form, position 0 pattern, init statistics, gradient flows to the table.
**Relevant Context:** `llm/embeddings.py`, `llm/positional.py`, `tests/test_embeddings.py`, `tests/test_positional.py`.
**Status:** [x] done (`llm/embeddings.py`, `llm/positional.py`, 18 tests pass). Notes: `TokenEmbedding(vocab, d_model, init, std, scale)` with `.weight`; `init_embedding_()` is reusable for other tables; positional modules expose `.encoding(seq_len)` and are built via `build_positional(kind, d_model, max_len)`; sinusoidal table is a non-persistent buffer, learned is a parameter.

---

### 4. Attention, multi-head attention, LayerNorm, feed-forward
**Intent:** Implement the heart of the transformer so each step is visible and inspectable.
**Expected Outcomes:** Scaled dot-product attention with causal mask returning both output and weights; `MultiHeadAttention` with configurable heads and `d_model` (validating divisibility), exposing Q/K/V and per-head weights; hand-written `LayerNorm` exposing mean/variance/normalized/scaled stages; `FeedForward` with GELU.
**Todo List:**
1. Implement scaled dot-product attention with optional causal mask and dropout; return weights.
2. Implement `MultiHeadAttention` (Q/K/V projections, head split/merge, output projection) with an optional "capture intermediates" mode.
3. Implement `LayerNorm` by hand with a stage-by-stage method for the educational breakdown.
4. Implement `FeedForward` (expand 4x, GELU, project back) and a residual/pre-norm helper.
5. Unit tests: attention rows sum to 1, causal mask zeroes future positions, MHA output shape, numerical match against `torch.nn.functional.scaled_dot_product_attention` and `nn.LayerNorm` (reference only inside tests), invalid head count raises.
**Relevant Context:** `llm/attention.py`, `llm/layers.py`, `tests/test_attention.py`, `tests/test_layers.py`.
**Status:** [x] done (`llm/attention.py`, `llm/layers.py`, 26 tests pass). Notes: `MultiHeadAttention(d_model, n_heads, dropout, causal, bias)`; `forward(x, return_details=True)` returns `(out, {q,k,v,scores,weights,head_output})` with shapes `(B,H,T,*)`; `LayerNorm.stages(x)` returns `{mean,var,normalized,output}`; `FeedForward(d_model, d_hidden, dropout, activation)` with `forward(x, return_hidden=True)`; helpers `causal_mask`, `attention_scores`, `scaled_dot_product_attention`, `gelu`, `relu`.

---

### 5. Full transformer (GPT) model and generation
**Intent:** Assemble components into a complete decoder-only model and enable text generation.
**Expected Outcomes:** `TransformerBlock` (pre-norm: LN -> MHA -> residual, LN -> FFN -> residual), `GPT` model with configurable layers/heads/d_model/context length/dropout, optional weight tying, a `forward` that can return per-layer intermediates, and `generate()` with temperature, top-k and top-p sampling; parameter counting helper.
**Todo List:**
1. Implement `TransformerBlock` and `GPT` with a config dataclass (validated).
2. Add intermediates capture (embeddings, per-layer attention weights, residual stream snapshots, logits) for visualization.
3. Implement `generate()` with temperature, top-k, top-p, and context cropping.
4. Add parameter-count breakdown per component.
5. Unit tests: output shape, loss on random init close to ln(vocab), causality (changing a future token does not change earlier logits), generation length and determinism with a seed, sampling-parameter edge cases.
**Relevant Context:** `llm/transformer.py`, `tests/test_transformer.py`.
**Status:** [x] done (`llm/transformer.py`, 28 tests; full suite 96 pass). Notes: `GPTConfig` (dataclass, validated; `to_dict`/`from_dict`); `GPT(cfg)` with `forward(ids, targets=None, return_details=False)` -> `logits` | `(logits, loss)` | `+ details` dict (`token_embeddings`, `with_position`, `blocks[...]`, `final_norm`, `logits`); `cross_entropy` is hand-written (also used as the training loss); `next_token_distribution(logits, temperature, top_k, top_p)` is reusable for the Inference page; `generate(ids, max_new_tokens, temperature, top_k, top_p, generator)`; `num_parameters()` and `parameter_breakdown()`.

---

### 6. Training pipeline
**Intent:** Provide a complete, understandable training loop with hand-written optimizers and scheduling.
**Expected Outcomes:** Data loading/splitting/batching; cross-entropy loss; Adam and SGD(momentum) hand-written (plus `torch.optim` option); warmup + cosine LR schedule; a trainer that yields per-step metrics (train loss, validation loss, LR, grad norm, tokens/sec) so any UI can stream it; checkpoint save/load.
**Todo List:**
1. `data.py`: load bundled corpus and `.txt` files from `input/` (recursive), train/val split, random batch sampler of (x, y) shifted by one token.
2. `optimizers.py`: Adam (bias correction, weight decay option) and SGD with momentum, following the `torch.optim.Optimizer` interface.
3. `schedulers.py`: linear warmup then cosine decay to a minimum LR; pure function of step.
4. `trainer.py`: training loop as a generator, gradient clipping, periodic evaluation, cooperative stop flag, device selection from config.
5. `checkpoint.py`: save/load model weights, optimizer state, model config, tokenizer state, metrics history; timestamped filenames into `output/`; validation of loaded files.
6. Add the bundled sample corpus under `data/` and record its source and license in `README.md`.
7. Unit tests: batch shapes and shift-by-one, Adam/SGD step equals `torch.optim` for several steps, scheduler boundary values, training reduces loss on a tiny overfit task, checkpoint round-trip gives identical logits, corrupted checkpoint handling.
**Relevant Context:** `training/*.py`, `data/`, `tests/test_optimizers.py`, `tests/test_schedulers.py`, `tests/test_trainer.py`, `tests/test_checkpoint.py`, `tests/test_data.py`.
**Status:** [x] done (`training/data.py`, `optimizers.py`, `schedulers.py`, `trainer.py`, `checkpoint.py`, `data/sample_corpus.txt`; test-gate OK, 158 tests pass). Notes for later sub-tasks: the bundled corpus is original text written for this project (no third-party license; mention in README). `build_corpus(use_bundled, use_input, input_dir)`, `encode_corpus`, `train_val_split`, `get_batch`; `TrainConfig` dataclass; `train(model, train_data, val_data, cfg, device, should_stop, optimizer, start_step)` is a generator yielding dicts with `step, train_loss, val_loss, lr, grad_norm, tokens_per_sec, elapsed, done`; `build_optimizer(name, ...)` names: adam, adamw, sgd, torch_adamw, torch_sgd; `build_schedule(name, base_lr, warmup, total, min_ratio)` names: warmup_cosine, constant; `save_checkpoint(model, tokenizer, train_config, history, optimizer, name, output_dir)`, `load_checkpoint(path, device)` -> dict(model, tokenizer, train_config, history, optimizer_state), `list_checkpoints()`, `CheckpointError`. Checkpoints load with `weights_only=True`.

---

### 7. Visualization layer
**Intent:** Convert model internals and metrics into Plotly figures, independent of Streamlit.
**Expected Outcomes:** Reusable figure builders returning `plotly.graph_objects.Figure`.
**Todo List:**
1. Tokenization: token-colored text view data, BPE merge-step table, vocabulary frequency chart.
2. Embeddings: heatmap of vectors, 2D projection (PCA via torch SVD), cosine similarity matrix.
3. Positional encoding: heatmap and per-dimension sine/cosine curves.
4. Attention: attention heatmap per head/layer, head comparison grid, Q/K/V matrix view.
5. LayerNorm: before/after distributions.
6. Training: live loss curves (train/val), LR curve, grad norm, throughput.
7. Data flow: pipeline diagram figure showing tensor shapes at each stage.
8. Unit tests: each builder returns a figure with expected traces for small inputs; handles empty/edge input.
**Relevant Context:** `viz/*.py`, `tests/test_viz.py`.
**Status:** [x] done (`viz/common.py`, `tokens.py`, `embeddings.py`, `attention.py`, `training.py`; test-gate OK, 184 tests pass). Notes for later sub-tasks: all builders return `plotly.graph_objects.Figure` (or HTML/rows for tokens) and show a centred message when given empty data. API: `viz.tokens` -> `tokens_html(tokens, ids)`, `token_rows`, `merge_rows(tokenizer)`, `vocab_frequency_figure`, `compression_figure`; `viz.embeddings` -> `embedding_heatmap`, `embedding_scatter` (PCA), `similarity_heatmap`, `positional_heatmap`, `positional_curves`, `pca_2d`, `cosine_similarity_matrix`; `viz.attention` -> `attention_heatmap(weights, tokens, pre_softmax=)`, `heads_grid(weights[H,T,T], tokens)`, `qkv_heatmap`, `attention_received_figure`, `layernorm_distribution`, `layernorm_token_stats`; `viz.training` -> `loss_figure(history, smooth_window, baseline, log_y)`, `lr_figure`, `grad_norm_figure`, `throughput_figure`, `schedule_figure(schedule, total_steps)`, `model_stages(cfg, batch, seq)`, `block_stages`, `dataflow_figure(stages)`, `parameter_breakdown_figure`, `moving_average`. test-gate requires one `test_<module>.py` per module, so viz tests are split across `test_common.py`, `test_tokens.py`, `test_viz_embeddings.py`, `test_viz_attention.py`, `test_training.py`.

---

### 8. Educational content layer
**Intent:** Keep explanations, formulas and live code snippets organized and reusable.
**Expected Outcomes:** Per-component content modules with intuitive explanations, LaTeX formulas (rendered with `st.latex`), and a helper that extracts the source of the real classes/functions for display next to demos.
**Todo List:**
1. Create `edu/` content modules for: tokenization, embeddings, positional encoding, attention, multi-head attention, LayerNorm, FFN, residual connections, full model, loss, Adam, LR scheduling, sampling.
2. Add a snippet helper using `inspect.getsource` so displayed code equals the executed code (comments in the source are the "detailed comments").
3. Write each explanation as: intuition, formula, shape table, common pitfalls.
4. Unit tests: snippet helper returns non-empty source for each referenced symbol; every content module exposes the expected fields.
**Relevant Context:** `edu/*.py`, `tests/test_edu.py`.
**Status:** [x] done (`edu/lessons.py`, `edu/snippets.py`; test-gate OK, 248 tests pass). Notes for sub-task 9: `edu.lessons.all_lessons()` returns 13 `Lesson` objects in teaching order (keys: tokenization, embeddings, positional, attention, multihead, layernorm, ffn, residual, model, loss, optimizer, schedule, sampling); each has `title, summary, intuition (markdown), steps, formulas (Formula.latex/.meaning -> st.latex), shapes (tensor, shape, meaning), pitfalls, snippets (dotted paths), related`. `edu.snippets.get_source(path)` returns the live, dedented source for `st.code(..., language="python")`; `get_docstring`, `line_count`, `resolve`; raises `SnippetError`. Tests verify every snippet path resolves, so renaming a symbol breaks a test instead of the UI.

---

### 9. Streamlit application and learning pages
**Intent:** Provide the interactive UI with clear navigation between components, parameter controls and immediate feedback.
**Expected Outcomes:** `app.py` with sidebar navigation and shared session state (config, tokenizer, model, metrics); one page per component with: explanation, formulas, interactive widget demo, visualization and code tab.
**Todo List:**
1. `app.py`: page config, navigation, shared session state, device info, global model-config sidebar.
2. Pages: Home/roadmap, Tokenization, Embeddings, Positional Encoding, Attention, Multi-Head Attention, LayerNorm and FFN, Full Transformer (data-flow explorer with shapes and parameter breakdown).
3. Each page: step-by-step walkthrough (stepper or tabs), sliders/selectors for hyperparameters (d_model, heads, layers, context, init, merges), live updates, "Explain / Math / Code" tabs.
4. Training Lab page: hyperparameter controls (optimizer, LR, warmup, batch size, steps, seed), Start/Stop, live Plotly loss and LR charts updated while training, metrics table, checkpoint save/load (files listed from `output/`).
5. Inference page: load current or saved model, prompt box, temperature/top-k/top-p/max tokens controls, generated text, optional per-token probability view and attention view of the generated sequence.
6. Handle empty-state and error cases (no model trained yet, invalid head/d_model combination, corrupted checkpoint) with clear messages.
7. Unit tests for non-UI glue logic (state initialization, config validation); smoke test of pages using `streamlit.testing.v1.AppTest`.
**Relevant Context:** `app.py`, `pages/`, `tests/test_app_smoke.py`.
**Status:** [x] done (test-gate OK, 402 tests pass). Built: `app.py` (st.navigation, 10 pages), `pages/page_{home,tokenization,embeddings,positional,attention,multihead,norm_ffn,transformer,training,inference}.py`, and the Streamlit-free `ui/state.py`, `ui/demo.py`, `ui/lab.py` plus the Streamlit helpers `ui/components.py` and `ui/sidebar.py`. Every page and module has its own `tests/test_*.py` (page tests use `AppTest` via `tests/conftest.py`, which redirects OUTPUT_DIR and INPUT_DIR to temp folders). Notes for sub-task 10: the app starts with `streamlit run app.py --server.port $APP_PORT`; `app.py` guards `main()` with `if __name__ == "__main__"`. `output/` must stay empty apart from `.gitkeep`. Old resume notes below are kept for history only.
Resume plan:
- 9a: `ui/state.py` (session-state defaults, config validation) and `ui/demo.py` (pure helpers that build small demo models and inputs), each with a `tests/test_<name>.py`.
- 9b: `ui/components.py` (Explain/Math/Code tabs from `Lesson`), `ui/sidebar.py` (model-config sidebar) and `app.py` (`st.navigation`).
- 9c: learning pages: home, tokenization, embeddings, positional, attention, multihead, norm_ffn, transformer.
- 9d: Training Lab and Inference pages.
- 9e: `AppTest` smoke tests, then test-gate.
Notes for resuming:
- test-gate needs a `test_<module>.py` for every `.py` file, including each `pages/*.py`. Keep page files thin and put the logic in `ui/` modules. For `pages/N_name.py`, the matching test must be `test_N_name.py`; alternatively use unnumbered names such as `pages/tokenization.py` with `tests/test_tokenization.py`, which already exists for the tokenizer, so avoid name clashes (e.g. `page_tokenization.py`).
- Do not pass empty arguments to `write_file`. Write files in chunks of about 200 lines or fewer.
- The bundled corpus has 4,876 characters and a 51-character vocabulary. BPE with 200 merges gives 251 tokens and trains in 0.14 s, so it is fast enough to retrain on every widget change.
- `tests/test_viz.py` was deleted externally; the split viz tests replace it.
- `training.trainer.train` yields dicts with `step, train_loss, val_loss, lr, grad_norm, tokens_per_sec, elapsed, done`. It raises `FloatingPointError` and `ValueError` (block_size larger than the context) for the UI to catch.
- Streamlit 1.67: use `st.plotly_chart(fig, width="stretch", key=...)`.

---

### 10. Scripts, documentation and compliance
**Intent:** Make the project runnable with one command and documented per the workspace rules.
**Expected Outcomes:** Executable `scripts/start.sh`, `stop.sh`, `cleanup.sh`; `README.md` (architecture + workflow Mermaid diagrams, License type, bundled data attribution); `Docs/Quickstart.md`; `Docs/Architecture.md`; all checks green.
**Todo List:**
1. Run `project-lifecycle-scripts` skill; configure start script to launch Streamlit detached on `APP_PORT` from `.env` (not 5000) and print the URL.
2. Run `project-docs` skill to produce README, Quickstart, Architecture; choose and document the open-source License.
3. Run `secure-config` scan for hard-coded secrets/URLs.
4. Run `test-gate`; must exit 0.
5. Run `project-audit`; fix findings.
6. Launch the app via `start.sh` and manually verify each page and a short training run, then stop it.
**Relevant Context:** `.bob/skills/project-lifecycle-scripts/`, `.bob/skills/project-docs/`, `.bob/skills/test-gate/`, `.bob/skills/project-audit/`, `.bob/skills/secure-config/`.
**Status:** [x] done - scripts/ (start, stop, cleanup), .env.example, README + Docs/Quickstart + Docs/Architecture (MIT), browser check (300-step run, inference), secrets scan, gitignore check, test-gate (402 passed) and project-audit all pass.

## Open Items / Risks to Confirm During Implementation
- License type for the project and the exact bundled corpus (must be public domain or permissively licensed; record attribution).
- Training in a Streamlit rerun model: use a generator-driven loop with `st.empty()` placeholders and a stop flag in `st.session_state`; verify responsiveness on CPU/MPS with the default tiny model.
- `.env` default for the llama.cpp URL is irrelevant to this app (no external LLM used); no network access required at runtime.

## Sub-task 11: Documentation of the ten sub-tasks
**Status:** [x] done. Per sub-task: scientific explanation (`Docs/subtasks/NN_slug_explanation.md` + `.html`) and implementation guide (source `Docs/subtasks/guide_src/NN_slug_guide.md`, page `NN_slug_implementation_guide.html`), plus `Docs/subtasks/index.html`. Generated by the `docsgen/` package (`scripts/build-docs.sh`, `--only N`); numbers come from live code and recorded runs (`{{fact:NAME}}`), pages are validated at build time, and `docsgen/` has its own unit tests in `tests/`.
