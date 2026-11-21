# Implementation guide: scripts, documents and compliance

## Role in the project

This sub-task wraps the application in everything needed to run it, trust it and hand it over: five lifecycle scripts, an environment file template, three project documents, a generator for the sub-task documentation, and a set of project-wide checks. None of it changes how the model behaves; it changes how the project is operated.

```flow Life of the project, from clone to delivery
 git clone
    |
 cp .env.example .env              optional: every variable has a default
    |
 uv venv .venv && uv pip install -r requirements.txt --python .venv/bin/python
    |
 scripts/start.sh  ---> .run/app.pid, .run/app.log, prints http://APP_HOST:APP_PORT
    |                         |
    |                  scripts/stop.sh   SIGTERM, wait, SIGKILL only if needed
    |
 edit code / text
    |
 scripts/build-docs.sh            Docs/subtasks/*.html  (validated)
    |
 bash .bob/skills/test-gate/scripts/run_tests.sh .        tests + "every module has a test"
 bash .bob/skills/secure-config/scripts/scan_secrets.sh . --strict
 bash .bob/skills/git-hygiene/scripts/ensure_gitignore.sh --check
 bash .bob/skills/project-audit/scripts/audit.sh .
    |
 scripts/cleanup.sh               remove .venv, __pycache__, .run
```

## File map

| Path | Role |
|---|---|
| `scripts/start.sh` | launch detached, print the URL |
| `scripts/stop.sh` | graceful shutdown |
| `scripts/cleanup.sh` | remove virtual environment, caches and runtime state (`--dry-run` lists only) |
| `scripts/build-docs.sh` | rebuild `Docs/subtasks/*.html` |
| `scripts/github-push.sh` | interactive helper that initialises a repository, sets or changes the remote, stages and pushes |
| `.env.example` | documented template of every environment variable |
| `config.py` | the only module that reads the environment |
| `README.md` | overview, architecture and workflow diagrams, quick start, layout, tests, data attribution, **License (MIT)** |
| `Docs/Quickstart.md` | install, configure, run, tour, tests, troubleshooting |
| `Docs/Architecture.md` | layers, model, training loop and UI data flow as Mermaid diagrams, design decisions |
| `Docs/subtasks/` | this documentation: 10 explanations (`.md` and `.html`), 10 implementation guides (`.html`), `index.html`, and the sources under `guide_src/` |
| `docsgen/` | the generator and validator behind `Docs/subtasks/` |
| `.gitignore` | policy managed by the git-hygiene skill |
| `.bob/skills/*/scripts/` | the check scripts used before delivery |
| `pytest.ini`, `tests/` | `testpaths = tests`, `pythonpath = .` |

## Configuration

{{file:.env.example|bash}}

{{source:config.load_settings}}

{{source:config.resolve_device}}

| Variable | Default | Validation | Consumer |
|---|---|---|---|
| `APP_PORT` | 8601 | integer; **5000 is rejected** | `config.load_settings`, `start.sh` |
| `APP_HOST` | `localhost` | none | `start.sh` (bind address and printed URL) |
| `APP_START_CMD` | `.venv/bin/python -m streamlit run app.py ...` | none | `start.sh` |
| `DEVICE` | `auto` | one of `auto`, `cpu`, `mps`, `cuda` | `config.resolve_device` |
| `DEFAULT_SEED` | 42 | integer | `ui.state.default_settings` |
| `INPUT_DIR` | `input` | resolved against the project root when relative | `training.data` |
| `OUTPUT_DIR` | `output` | resolved against the project root when relative | `training.checkpoint` |

`config.py` loads `.env` with `override=False`, so a variable already set in the shell wins over the file. Settings are returned as a frozen dataclass.

!!! note "Why `APP_START_CMD` is single-quoted"
    The value contains `$APP_PORT` and `$APP_HOST`. Single quotes keep them unexpanded when the file is read, and `start.sh` expands them when it runs the command, after the final port and host are known.

## Lifecycle scripts

### start.sh

{{file:scripts/start.sh|bash}}

```flow start.sh decisions
load .env (set -a; . ./.env; set +a)      sourced, so keep it free of commands
APP_HOST, APP_PORT, APP_START_CMD         defaults if unset
port 5000 on Darwin                       exit 1
pid file exists and process alive         print URL, exit 0 (idempotent)
nohup bash -c "$APP_START_CMD" > .run/app.log &
write pid; sleep 1
process died                              print log path, remove pid file, exit 1
otherwise                                 print pid, URL, log path
```

### stop.sh

{{file:scripts/stop.sh|bash}}

The stop timeout is read from `STOP_TIMEOUT` (default 15 seconds). Children are signalled first with `pkill -P`, then the parent, so the Streamlit worker does not outlive the launcher.

### cleanup.sh

{{file:scripts/cleanup.sh|bash}}

It calls `stop.sh` first (unless `--dry-run`), then removes only the four named kinds of folders, never touching `.git`.

### build-docs.sh

{{file:scripts/build-docs.sh|bash}}

Pass `--only 6 7` to rebuild some sub-tasks. A partial build skips the cross-page link check because the index links to all 20 pages.

All five scripts must be executable (`chmod +x scripts/*.sh`); the audit checks this.

## The documentation generator

`docsgen/` has no entry in the application. It is a build tool, and it imports the project only to read facts and source.

| File | Lines | Role |
|---|---|---|
| `docsgen/catalog.py` | 81 | the ten `SubTask` records and the file names of their deliverables |
| `docsgen/facts.py` | 150 | computed values used by the fact directive: live (from the model and corpus) and recorded (from two JSON files of measured runs) |
| `docsgen/render.py` | 305 | Markdown dialect to HTML: math to MathML, directives, flow figures, admonitions, table of contents |
| `docsgen/charts.py` | 241 | inline SVG charts of the recorded reference run, each with a data table |
| `docsgen/theme.py` | 209 | one embedded stylesheet with light and dark code themes |
| `docsgen/page.py` | 117 | the page frame: header, sub-navigation, pipeline, table of contents, footer |
| `docsgen/validate.py` | 149 | structural and accessibility checks on the HTML |
| `docsgen/build.py` | 149 | builds the 20 pages and the index, then validates them |
| `docsgen/reference_run.json`, `long_runs.json` | | recorded training histories used for the charts and the quoted measurements |

### Build flow

```flow python -m docsgen.build
catalog.SUBTASKS ──for each sub-task──┐
   NN_slug_explanation.md             │  render(md)  ── expand fact directives  ── extract code fences and math (stash tokens)
   guide_src/NN_slug_guide.md         │             ── Markdown to HTML   ── restore stashed parts, run directives
                                      ▼
                          page.build_page(meta, rendered)   one standalone HTML string
                                      │
        NN_slug_explanation.html      │
        NN_slug_implementation_guide.html
                                      ▼
   index.html (from index_markdown)   validate.check_file(path, check_links=not partial)
                                      │
            problems? ──yes──▶ BuildError, exit 1        no ──▶ "wrote Docs/subtasks/..." for every file
```

### The Markdown dialect

| Syntax | Result |
|---|---|
| `$...$` and `$$...$$` | inline and display MathML (via latex2mathml) |
| fact directive: two opening braces, `fact:`, a name, two closing braces | value from `docsgen.facts`; an unknown name is a build error |
| source directive: `source:` and a dotted path inside double braces, alone on a line | live source of a symbol, with a caption of file and line numbers |
| file directive: `file:` and `path\|lang\|first-last` inside double braces | excerpt of a repository file |
| chart directive with the name `loss`, `lr` or `gradnorm` | inline SVG chart and a data table |
| a fence starting with `flow` | captioned ASCII figure, keyboard focusable |
| other fences | highlighted with Pygments |
| `!!! note "Title"` followed by indented text | admonition |
| `[text](NN_slug_kind.md)` | rewritten to `.html` |

{{source:docsgen.render.source_figure}}

{{source:docsgen.render.file_figure}}

A directive that fails (a path that does not resolve, a missing file, a bad line range, an unknown fact) raises `RenderError`, and the build stops with the message. A document therefore cannot silently show a stale or empty listing.

### Validation

{{source:docsgen.validate.check_html}}

| Check | Why |
|---|---|
| doctype, `lang="en"`, `<title>`, viewport meta | basic page validity and screen-reader language |
| one `<main>`, a skip link | landmark and keyboard navigation |
| exactly one `h1`, no skipped heading level | logical outline |
| unique ids, every `#anchor` exists | working in-page links |
| no `<script>`, no `url(`, no external `src` or `href` to assets | standalone and offline |
| every `<svg>` has `role="img"` and a `<title>` | text alternative |
| every `<pre>` has `tabindex` or an `aria-label` | scrollable code reachable by keyboard |
| no unreplaced `{{...}}` and no internal stash tokens | rendering errors |
| links to sibling `.html` files exist (full build) | no dead cross-references |

## Compliance checks

| Script | Checks | Exit |
|---|---|---|
| `test-gate/scripts/run_tests.sh .` | runs pytest, then reports any Python module without a `test_<module>.py` | 0 pass, 1 failures or untested code, 2 no tests found |
| `secure-config/scripts/scan_secrets.sh . --strict` | quoted literals assigned to `api_key`, `secret`, `token`, `password`; token shapes; private keys; e-mail addresses in non-Markdown files; hard-coded `http(s)://` URLs in strict mode; `.env.example` consistency. A line with the word `nosecret` is exempt | 0 clean, 1 findings |
| `git-hygiene/scripts/ensure_gitignore.sh --check` | every required ignore entry is present; without `--check` it adds the missing ones and creates `.gitkeep` files in `input/` and `output/` | 0 complete, 1 missing |
| `project-audit/scripts/audit.sh .` | README present, a License section, Quickstart, Architecture, documents only in `Docs/` (the root may hold only README, AGENTS, CLAUDE and LICENSE), ignore policy, secret scan, executable scripts, container policy if a container file exists | 0 passed, 1 failed |

The audit does not run the tests; the test gate does.

### Test layout

| Group | Files | Count |
|---|---|---|
| Model | `test_tokenizer` 19, `test_embeddings` 9, `test_positional` 9, `test_attention` 16, `test_layers` 10, `test_transformer` 28 | 91 |
| Training | `test_data` 11, `test_optimizers` 15, `test_schedulers` 9, `test_trainer` 18, `test_checkpoint` 9 | 62 |
| Visualisation and content | `test_common` 4, `test_tokens` 4, `test_viz_embeddings` 6, `test_viz_attention` 4, `test_lessons` 54, `test_snippets` 10, `test_components` 7, `test_training` 8 (the figure builders of `viz/training.py`) | 97 |
| Application | `test_app` 8, `test_state` 25, `test_sidebar` 3, `test_lab` 19, `test_demo` 32, `test_config` 5, ten `test_page_*` files (60 in total) | 152 |

The four groups add up to the 402 tests of the project before the documentation generator. The documentation generator has its own test files, `tests/test_catalog.py`, `test_theme.py`, `test_charts.py`, `test_facts.py`, `test_render.py`, `test_page.py`, `test_validate.py` and `test_build.py`, so that every module of `docsgen/` also has a test module. The exact per-file counts are printed by `pytest --collect-only -q`.

## Integration points

| Where | Interaction |
|---|---|
| `scripts/start.sh` | reads `.env`; runs `APP_START_CMD`, which starts `app.py` |
| `config.py` | the single reader of the environment for application code |
| `docsgen.facts` | imports the live model, tokenizer and training data to compute numbers |
| `docsgen.render` | reuses `edu.snippets` to read source |
| `.bob/skills/*` | the checks are separate scripts, called by hand or by a pipeline |

## Runtime behaviour

- `start.sh` writes `.run/app.pid` and `.run/app.log`; `.run/` is ignored by git and removed by `cleanup.sh`.
- Running `start.sh` twice reports the existing process instead of starting another.
- `stop.sh` on a stale pid file removes it and exits 0.
- `build-docs.sh` needs the project virtual environment because it imports the model code; the global Python has no PyTorch.
- `output/` should hold only `.gitkeep` when the project is delivered; the tests redirect their checkpoints to temporary folders.

## Common pitfalls

- **Port 5000.** Refused on purpose.
- **A dollar sign in prose of a Markdown source.** A single `$` can start a formula. Write amounts in words or inside code spans.
- **A new source module without a test module.** The test gate fails with `UNTESTED`.
- **Editing generated `.html` by hand.** The next build overwrites it; edit the Markdown source instead.
- **A document placed in the project root.** The audit allows only the four named files there.
