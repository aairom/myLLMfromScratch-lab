# Implementation guide: project scaffolding and environment

## Role in the project

This sub-task creates the ground every other module stands on. It produces no model code; it produces the *rules of the workspace*: where files go, how configuration is read, how tests run, and what is kept out of Git.

```flow Where sub-task 1 sits in the build order
[1 scaffolding] -> [2 tokenization] -> [3 embeddings + positions] -> [4 attention layers]
        |                                                                   |
        |                                                                   v
        |                                                          [5 GPT model + sampling]
        |                                                                   |
        |                                                                   v
        +---- config, tests, scripts, .gitignore --------------> [6 training pipeline]
                                                                            |
                                              [7 viz] -> [8 lessons] -> [9 Streamlit app] -> [10 scripts + docs]
```

## Repository layout

```flow Top-level layout (generated folders omitted)
myLLMfromScratch-lab/
|-- app.py                 Streamlit entry point (sub-task 9)
|-- config.py              settings from the environment (this sub-task)
|-- llm/                   tokenizer, embeddings, attention, GPT   (sub-tasks 2-5)
|-- training/              data, optimizers, schedules, trainer    (sub-task 6)
|-- viz/                   Plotly figure builders                  (sub-task 7)
|-- edu/                   lessons and live code snippets          (sub-task 8)
|-- ui/  pages/           shared state and the ten pages          (sub-task 9)
|-- data/sample_corpus.txt bundled demonstration text
|-- input/   output/       your texts in, checkpoints out (contents are git-ignored)
|-- scripts/               start.sh stop.sh cleanup.sh build-docs.sh  (sub-task 10)
|-- tests/                 one test_<module>.py per module
|-- Docs/                  Quickstart, Architecture and these sub-task documents
|-- .env.example           every configurable variable, documented
|-- requirements.txt       dependencies (installed with uv into .venv)
`-- pytest.ini             testpaths = tests, pythonpath = .
```

## Configuration

All configuration is read by one module. No other file contains a port, a path or a seed literal.

{{source:config.Settings}}

{{source:config.load_settings}}

### Environment variables

| Variable | Default | Meaning | Validation |
|---|---|---|---|
| `APP_PORT` | `8601` | Port of the Streamlit app | integer; `5000` is rejected (reserved by AirDrop on macOS) |
| `APP_HOST` | `localhost` | Address the app binds to and prints | read by `scripts/start.sh` |
| `APP_START_CMD` | `.venv/bin/python -m streamlit run app.py ...` | Command launched by `start.sh` | read by `scripts/start.sh` |
| `DEVICE` | `auto` | `auto`, `cpu`, `mps` or `cuda` | other values raise `ValueError` |
| `DEFAULT_SEED` | `42` | Seed shown in the sidebar and used for demos | integer |
| `INPUT_DIR` | `input` | Folder with optional `.txt` and `.md` training files | relative paths resolve against the project root |
| `OUTPUT_DIR` | `output` | Folder for checkpoints | relative paths resolve against the project root |

`config.py` calls `load_dotenv(PROJECT_ROOT / ".env", override=False)`, so a real environment variable always wins over the file. `.env.example` is committed; `.env` is ignored.

### Device selection

{{source:config.resolve_device}}

`auto` prefers CUDA, then Apple MPS, then CPU. An explicit choice that is not available falls back to CPU, so the application always starts.

## Data flow of configuration

```flow From .env to a running component
.env.example --(copy, edit)--> .env
                                  |
                  python-dotenv (override=False)
                                  v
 process environment --> config.load_settings() --> Settings (frozen dataclass)
                                                        |
        +-------------------+---------------+-----------+--------------+
        v                   v               v                          v
   app_port            device          default_seed              input_dir / output_dir
 (start.sh reads     resolve_device   ModelSettings.seed        training.data / training.checkpoint
  APP_PORT itself)   -> torch.device  (sidebar default)         (read recursively / write timestamped)
```

## Data structures

```flow Settings and the seeds derived from it
Settings
  app_port     : int      8601 unless overridden
  device       : str      "auto" | "cpu" | "mps" | "cuda"
  default_seed : int      42
  input_dir    : Path     <project>/input
  output_dir   : Path     <project>/output

Seeds used at run time (three independent places)
  model weights    torch.manual_seed(ModelSettings.seed)        in ui.state.build_model
  training batches torch.Generator().manual_seed(cfg.seed + start_step)   in training.trainer.train
  evaluation       torch.Generator().manual_seed(seed)  fixed   in training.trainer.estimate_loss
```

## Tests

`pytest.ini` points pytest at `tests/` and adds the project root to the import path, so modules are imported by their package names.

{{file:pytest.ini|ini}}

Streamlit pages are tested headlessly with `AppTest`. The shared fixture redirects `OUTPUT_DIR` and `INPUT_DIR` into a temporary folder, so a test run never writes to `output/` and never reads your own files from `input/`.

{{file:tests/conftest.py|python|19-24}}

| Fact | Value |
|---|---|
| Unit tests in the application suite | 402 |
| Rule | every `*.py` module has a `test_<module>.py` |
| Gate | `bash .bob/skills/test-gate/scripts/run_tests.sh .` must exit 0 |

## Environment and dependencies

```bash
uv venv                                  # creates .venv
uv pip install -r requirements.txt       # installs the declared dependencies
.venv/bin/python -m pytest -q            # runs the tests with that interpreter
```

The application and its tests must run with `.venv/bin/python`: a system-wide Streamlit does not have PyTorch installed. After any change to `requirements.txt`, run the install command again.

| Package | Used for |
|---|---|
| `torch` | tensors, autograd, `nn.Module` |
| `streamlit` | the browser interface |
| `plotly` | interactive figures |
| `numpy` | numerical helpers in the visualisation layer |
| `python-dotenv` | loading `.env` |
| `pytest` | the test suite |
| `markdown`, `latex2mathml`, `pygments` | only the documentation builder in `docsgen/` |

## Git hygiene

The `.gitignore` keeps the following out of version control: `.env` files (but not `.env.example`), `.venv/`, `__pycache__/`, `.run/`, archives and native libraries, `AGENTS.md`, `.bob/mcp.json`, and the **contents** of `input/` and `output/`. Both folders stay in the repository through a `.gitkeep` file.

## Runtime behaviour

- Importing `config` loads `.env` once. Changing `.env` needs an application restart.
- `load_settings()` raises `ValueError` immediately for port 5000 or an unknown device; the app fails fast instead of binding to a forbidden port.
- `output/` holds only `.gitkeep` until a checkpoint is saved; checkpoints are named `YYYYmmdd_HHMMSS_<name>.pt`.

## Integration points

| Consumer | What it takes from this sub-task |
|---|---|
| `ui.state` | `config.load_settings()` for the default seed and `resolve_device` for the device |
| `training.data` | `input_dir` for the recursive read of `.txt` and `.md` files |
| `training.checkpoint` | `output_dir` for saved checkpoints |
| `scripts/start.sh` | `APP_HOST`, `APP_PORT`, `APP_START_CMD` from `.env` |
| `tests/conftest.py` | redirects `INPUT_DIR` and `OUTPUT_DIR` |

## Common pitfalls

- **Wrong interpreter.** Running `streamlit run app.py` with a global Streamlit fails on `import torch`. Use `.venv/bin/python -m streamlit`, or `scripts/start.sh`.
- **Port 5000.** On macOS it belongs to AirDrop; the configuration rejects it by design.
- **Stale environment.** If imports fail after pulling changes, re-run `uv pip install -r requirements.txt`.
- **Checkpoints in Git.** Never remove the `output/*` rule: model files are binary and large.
