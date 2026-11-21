# Implementation guide: the educational content layer

## Role in the project

`edu/` holds everything a learner reads that is not a chart: the text, the formulas, the tensor shapes and the pointers to live code. It has no Streamlit import and no model state, so it can be tested as plain data. `ui/components.py` turns a lesson into three Streamlit tabs, and the pages call it with one line.

```flow From lesson data to the screen
 edu/lessons.py            _LESSONS (13 frozen Lesson records)     LESSONS dict, get_lesson(key), all_lessons()
        |                                                   
        |  lesson.snippets = ("llm.attention.causal_mask", ...)    dotted paths, not copied code
        v
 edu/snippets.py           resolve(path) -> object   get_source(path) -> str (inspect.getsource, dedented)
        |
        v
 ui/components.py          lesson_tabs(key, extra_sources)
        |-- Explain tab   render_explain(lesson)   intuition, numbered steps, pitfalls expander
        |-- Math tab      render_math(lesson)      st.latex(formula) + meaning, shape table
        '-- Code tab      render_code(lesson)      st.code(source) per snippet, first expanded
```

## Module map

| File | Lines | Contents |
|---|---|---|
| `edu/lessons.py` | 517 | `Formula`, `Lesson`, the 13 lessons, `LESSONS`, `get_lesson`, `all_lessons` |
| `edu/snippets.py` | 64 | `SnippetError`, `resolve`, `get_source`, `get_docstring`, `line_count` |
| `ui/components.py` | 98 | `init_page`, `plot`, `int_slider`, `lesson_tabs`, `render_explain`, `render_math`, `render_code`, `page_header`, `empty_state` |
| `tests/` | | `test_lessons.py` 54 (parametrised over the 13 lessons), `test_snippets.py` 10, `test_components.py` 7 |

## Data structures

{{source:edu.lessons.Formula}}

{{source:edu.lessons.Lesson}}

```flow Shape of the content (totals over the 13 lessons)
Lesson ─┬─ key, title, summary, intuition (Markdown)
        ├─ steps     tuple[str]                    57 sentences, 3 to 6 per lesson
        ├─ formulas  tuple[Formula(latex, meaning)] 34 formulas
        ├─ shapes    tuple[(tensor, shape, meaning)]
        ├─ pitfalls  tuple[str]                    24 items, at least 1 per lesson
        ├─ snippets  tuple[dotted path]            26 paths, 638 source lines
        └─ related   tuple[key]                    other lesson keys
```

Teaching order, which `all_lessons()` preserves:

| # | Key | Steps | Formulas | Pitfalls | Snippets |
|---|---|---|---|---|---|
| 1 | `tokenization` | 6 | 2 | 3 | 3 |
| 2 | `embeddings` | 4 | 2 | 2 | 2 |
| 3 | `positional` | 3 | 3 | 2 | 2 |
| 4 | `attention` | 6 | 3 | 3 | 3 |
| 5 | `multihead` | 5 | 3 | 2 | 3 |
| 6 | `layernorm` | 4 | 3 | 2 | 1 |
| 7 | `ffn` | 4 | 2 | 1 | 2 |
| 8 | `residual` | 4 | 3 | 1 | 1 |
| 9 | `model` | 5 | 2 | 1 | 2 |
| 10 | `loss` | 3 | 3 | 2 | 1 |
| 11 | `optimizer` | 5 | 4 | 2 | 3 |
| 12 | `schedule` | 3 | 2 | 1 | 1 |
| 13 | `sampling` | 5 | 2 | 2 | 2 |

### Lookup

{{source:edu.lessons.get_lesson}}

`get_lesson` raises a `KeyError` whose message lists the valid keys, so a typo in a page is diagnosed immediately.

## The snippet resolver

{{source:edu.snippets.resolve}}

{{source:edu.snippets.get_source}}

```flow resolve("llm.transformer.GPT.generate")
parts = ["llm", "transformer", "GPT", "generate"]
try import "llm.transformer.GPT"     ModuleNotFoundError -> shorter prefix
try import "llm.transformer"         ok  -> obj = module, attrs = ["GPT", "generate"]
getattr(obj, "GPT") -> class         getattr(class, "generate") -> function
get_source -> textwrap.dedent(inspect.getsource(function)).rstrip() + "\n"
```

| Input | Result |
|---|---|
| a function, class or method | its dedented source, comments and docstring included |
| a module path | `SnippetError` ("point to a class or function") |
| a path with fewer than two parts | `SnippetError` |
| a name that does not exist | `SnippetError` naming the missing attribute |
| a built-in without source | `SnippetError` ("source unavailable") |

`SnippetError` subclasses `LookupError`, so callers can catch either. `get_docstring` returns the cleaned docstring and `line_count` the number of source lines.

!!! note "Why the longest module prefix is tried first"
    A path such as `llm.transformer.GPT` could be read as module `llm`, attribute `transformer`. Trying the longest importable prefix first resolves it as module `llm.transformer` with attribute `GPT`, which is what the author meant.

## Rendering

{{source:ui.components.lesson_tabs}}

{{source:ui.components.render_code}}

The Code tab catches `SnippetError`, shows a `st.warning` that names the path and continues, so a broken snippet never breaks a page. The first snippet of a lesson is expanded and the others are collapsed.

### Which pages show which lesson

| Page | Lesson | Extra sources shown after the lesson's own |
|---|---|---|
| `page_tokenization.py` | `tokenization` | `llm.tokenizer.BPETokenizer.train` |
| `page_embeddings.py` | `embeddings` | |
| `page_positional.py` | `positional` | `llm.positional.sinusoidal_table` |
| `page_attention.py` | `attention` | `llm.attention.scaled_dot_product_attention` |
| `page_multihead.py` | `multihead` | `llm.attention.MultiHeadAttention` |
| `page_norm_ffn.py` | `layernorm` and `ffn` | |
| `page_transformer.py` | `model` | `llm.transformer.TransformerBlock`, `llm.transformer.GPT.forward` |
| `page_training.py` | `optimizer` | `training.trainer.train`, `training.optimizers.Adam` |
| `page_inference.py` | `sampling` | `llm.transformer.next_token_distribution`, `llm.transformer.GPT.generate` |
| `page_home.py` | all 13, as title and summary lines | |

The `residual`, `loss` and `schedule` lessons have no page of their own with tabs: they appear as a line on the home page. Their snippets and formulas are still validated by the tests.

## Writing a new lesson

1. Add a `Lesson(...)` to `_LESSONS` at the position in the teaching order where it belongs. The key must be unique.
2. Provide every section: an intuition of more than 100 characters, at least three steps, at least one formula, one shape row, one pitfall and one snippet path.
3. Write LaTeX without dollar signs (it is passed to `st.latex`), with balanced braces and matching `\left` and `\right`.
4. Add the key to `EXPECTED_KEYS` in `tests/test_lessons.py`.
5. Call `page_header(key)` and `lesson_tabs(key)` from a page.

## Integration points

| Where | Interaction |
|---|---|
| `pages/page_*.py` | `page_header` and `lesson_tabs` |
| `pages/page_home.py` | iterates `all_lessons()` for the learning path |
| `docsgen.render` | reuses `edu.snippets` for the `{{source:...}}` directive of this documentation |
| `llm`, `training`, `ui` | the lesson text refers to them by dotted path only; they do not import `edu` |

## Configuration and environment

No variable and no file is read. The content is in the Python source, and the only runtime input is the lesson key.

## Runtime behaviour

- `LESSONS` is built once at import time. Lookups are dictionary reads.
- Source is read through `inspect` on every call; the cost is a file read, which is negligible for 26 snippets.
- If a source file is edited while the app runs, the next page reload shows the new code only after Python re-imports the module (Streamlit reloads modules on a script change).
- Unknown keys raise `KeyError`; unknown snippet paths raise `SnippetError`, which the Code tab turns into a warning.

## Tests

`test_lessons.py` checks that exactly the 13 expected keys exist in teaching order, that every lesson has every section, that every snippet path returns non-empty source, that related keys exist and never include the lesson itself, that LaTeX braces and `\left` and `\right` are balanced and contain no dollar signs, and that `get_lesson` reports a helpful error. `test_snippets.py` covers classes, functions, methods, preserved comments, bad paths and built-ins. `test_components.py` drives the real components with Streamlit's `AppTest`: each of the 13 lessons renders the tabs Explain, Math and Code, and a missing snippet produces a warning.

## Common pitfalls

- **Forgetting that the prose is not checked against the code.** Update the lesson in the same change as the behaviour.
- **Dollar signs in a formula.** They are rejected by a test because `st.latex` expects raw LaTeX.
- **Renaming a function without searching the lessons.** A stale path is caught by the snippet test, but only if the test suite is run.
- **Expecting a page for every lesson.** Three lessons are shown only on the home page.
