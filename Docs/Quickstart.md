# Quickstart

## Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/) for the virtual environment
- About 1 GB of disk for PyTorch. No GPU is needed; Apple MPS is used
  automatically when available.

## 1. Install

```bash
uv venv .venv
uv pip install -r requirements.txt --python .venv/bin/python
```

After changing `requirements.txt`, run the second command again.

## 2. Configure (optional)

```bash
cp .env.example .env
```

| Variable | Default | Meaning |
|---|---|---|
| `APP_HOST` | `localhost` | Interface the web UI binds to |
| `APP_PORT` | `8601` | Port of the web UI (never 5000, AirDrop uses it) |
| `APP_START_CMD` | `.venv/bin/python -m streamlit run app.py ...` | Launch command used by `start.sh` |
| `DEVICE` | `auto` | `auto`, `cpu`, `mps` or `cuda` |
| `DEFAULT_SEED` | `42` | Seed for reproducible runs |
| `INPUT_DIR` | `input` | Extra training text (`.txt`, `.md`, read recursively) |
| `OUTPUT_DIR` | `output` | Where checkpoints are written |

## 3. Run

```bash
scripts/start.sh     # starts detached and prints the URL
scripts/stop.sh      # stops it
scripts/cleanup.sh   # removes .venv and __pycache__ when you are done
```

Open `http://localhost:8601`.

## 4. A first tour

1. **Home**: see the model summary in the sidebar.
2. **1. Tokenization**: type text and compare character and BPE splits.
3. **2-7**: follow the lessons; use the *Explain*, *Math* and *Code* tabs.
4. **8. Training lab**: keep the defaults and press **Start (from scratch)**.
   Loss should fall from about `ln(vocabulary)` (3.9 for the demo corpus) to
   about 3.0 within 300 steps. Train for 2000+ steps with a larger model to get
   readable text.
5. **9. Inference**: generate text and inspect token probabilities and
   attention. Try temperature 0 (greedy) against 1.2.

## 5. Use your own text

Drop `.txt` or `.md` files into `input/`, enable *Files from input/* in the
sidebar and press **Apply and re-initialise**.

## 6. Read or rebuild the sub-task documentation

Open `Docs/subtasks/index.html` in a browser. It links the scientific
explanation and the implementation guide of each of the ten sub-tasks. After
editing a Markdown source in `Docs/subtasks/`, run `scripts/build-docs.sh`
(`--only N` rebuilds one sub-task).

## 7. Run the tests

```bash
bash .bob/skills/test-gate/scripts/run_tests.sh .
```

## Troubleshooting

- **Port already in use**: change `APP_PORT` in `.env`.
- **`ModuleNotFoundError: torch`**: the app must be launched with
  `.venv/bin/python -m streamlit`, which `scripts/start.sh` already does.
- **Loss stays near 4.0**: raise the learning rate or the number of steps, and
  keep warm-up shorter than the total steps.
