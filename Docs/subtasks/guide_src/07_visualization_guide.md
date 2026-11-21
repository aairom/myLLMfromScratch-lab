# Implementation guide: the visualization layer

## Role in the project

`viz/` is the only place where Plotly figures are made. Each function is **pure**: it takes tensors, lists or a metrics history and returns a `plotly.graph_objects.Figure`. No function imports Streamlit, and only `viz/training.py` imports model code (the `GPTConfig` dataclass, to derive shapes). The pages call a builder and hand the result to `ui.components.plot`, which draws it.

```flow Where a figure comes from
 model / tokenizer / history
        |
        v
 ui.demo.*  or  state["history"]        compute the tensors (no drawing)
        |
        v
 viz.<module>.<builder>(...)            pure function -> go.Figure
        |
        v
 ui.components.plot(fig, key)           st.plotly_chart(fig, width="stretch", key=key)
```

## Module map

| File | Lines | Builders and helpers |
|---|---|---|
| `viz/common.py` | 80 | `to_numpy`, `display_token`, `unique_labels`, `empty_figure`, `heatmap` |
| `viz/tokens.py` | 82 | `tokens_html`, `token_rows`, `merge_rows`, `vocab_frequency_figure`, `compression_figure` |
| `viz/embeddings.py` | 119 | `pca_2d`, `cosine_similarity_matrix`, `embedding_heatmap`, `embedding_scatter`, `similarity_heatmap`, `positional_heatmap`, `positional_curves` |
| `viz/attention.py` | 123 | `attention_heatmap`, `heads_grid`, `qkv_heatmap`, `attention_received_figure`, `layernorm_distribution`, `layernorm_token_stats` |
| `viz/training.py` | 182 | `moving_average`, `loss_figure`, `lr_figure`, `grad_norm_figure`, `throughput_figure`, `schedule_figure`, `model_stages`, `block_stages`, `dataflow_figure`, `parameter_breakdown_figure` |
| `tests/` | | `test_common.py` 4, `test_tokens.py` 4, `test_viz_embeddings.py` 6, `test_viz_attention.py` 4, `test_training.py` 8 |

## Shared helpers

{{source:viz.common.to_numpy}}

`to_numpy` is the single entry point for every matrix. It accepts a tensor or a list, detaches it, moves it to the CPU and converts it to `float64`. Non-finite values (`-inf` from the causal mask, `NaN`) become `NaN`, which Plotly draws as an empty cell. That is how masked attention scores appear blank.

{{source:viz.common.unique_labels}}

{{source:viz.common.heatmap}}

```flow Defaults chosen by common.heatmap
colorscale   "RdBu"      diverging palette
zmid         0.0         zero is the neutral colour
y axis       reversed    row 0 at the top, like a matrix
wrong rank   -> empty_figure(title)     input must be 2-D and non-empty
```

{{source:viz.common.empty_figure}}

### Contract: figures never raise for missing data

| Situation | Result |
|---|---|
| empty list, empty tensor, wrong number of dimensions | a figure with no traces and one centred annotation |
| `NaN` or `inf` cells | blank cells |
| repeated tokens | distinct axis labels (`0·e`, `4·e`) |
| whitespace tokens | visible stand-ins (`␣`, `⏎`, `⇥`, `␍`) |

Tests recognise an empty message figure as one with zero traces and exactly one annotation.

## Embedding figures

{{source:viz.embeddings.pca_2d}}

```flow pca_2d(matrix)  matrix: (N, D)
centred = matrix - mean over rows
        |  torch.linalg.svd(centred, full_matrices=False)  ->  U, s, Vh
        v
coords    = centred @ Vh[:2].T            (N, 2)
explained = s[:2]**2 / sum(s**2)          (2,)
degenerate input (N < 2, D < 1, identical rows)  ->  zeros, zeros
```

{{source:viz.embeddings.cosine_similarity_matrix}}

{{source:viz.embeddings.embedding_scatter}}

| Builder | Input | Colour range | Notes |
|---|---|---|---|
| `embedding_heatmap(vectors, labels, max_dims=64)` | `(N, D)` | diverging, zero-centred | shows the first `max_dims` columns and says so in the title |
| `embedding_scatter(matrix, labels)` | `(N, D)` | | title reports the explained variance, axes show each component's share |
| `similarity_heatmap(matrix, labels)` | `(N, D)` | pinned to [-1, 1] | cosine matrix, diagonal 1 |
| `positional_heatmap(table)` | `(T, D)` | pinned to [-1, 1] | the sinusoidal table fits that range exactly |
| `positional_curves(table, dims=None)` | `(T, D)` | | one line per selected dimension, even index labelled sin and odd cos |

## Attention and normalisation figures

{{source:viz.attention.attention_heatmap}}

{{source:viz.attention.heads_grid}}

| Builder | Input | Notes |
|---|---|---|
| `attention_heatmap(weights, tokens, pre_softmax=False)` | `(T, T)` | probabilities use `Blues` pinned to [0, 1]; raw scores use `RdBu` centred on 0 |
| `heads_grid(weights, tokens, max_cols=4)` | `(H, T, T)` | small multiples; grid size `ceil(H / cols)` rows; one shared colour bar |
| `qkv_heatmap(matrix, tokens, name, head=None)` | `(T, d_k)` | rows are tokens, columns are dimensions inside the head |
| `attention_received_figure(weights, tokens)` | `(T, T)` or `(H, T, T)` | column mean of the weights, heads averaged first |
| `layernorm_distribution(before, after)` | any shape | two overlaid histograms, 40 bins |
| `layernorm_token_stats(before, after, tokens)` | two `(T, D)` | per-token mean and standard deviation; shapes must match |

## Token figures

{{source:viz.tokens.tokens_html}}

`tokens_html` is the only builder that returns **HTML** instead of a figure. Every token label is passed through `html.escape` before it is placed in a coloured chip, so user text can never inject markup; the page renders it with `unsafe_allow_html` for that reason only. The chip title attribute shows the token id.

{{source:viz.tokens.merge_rows}}

`merge_rows` reads `tokenizer.trace`. A tokenizer restored from a checkpoint has an empty trace (`BPETokenizer.to_dict` does not store it), so the merge table is empty after loading a checkpoint.

## Training figures

{{source:viz.training.moving_average}}

{{source:viz.training.loss_figure}}

```flow Metrics history -> figure
history = [ {step, train_loss, val_loss, lr, grad_norm, tokens_per_sec, ...}, ... ]
     |  _series(history, key): keep rows where the value is present and finite
     v
 (steps, values)  ->  go.Scatter
loss_figure: raw train line (opacity 0.35 when smoothed) + moving average + validation markers
             optional dotted line at ln(V); optional log y axis
lr_figure / grad_norm_figure / throughput_figure -> one line each, empty message before training
```

## Data-flow figures

{{source:viz.training.model_stages}}

{{source:viz.training.dataflow_figure}}

`dataflow_figure` draws one rectangle shape and one annotation per stage, plus an arrow annotation between neighbouring stages, so a pipeline of n stages has n shapes and 2n - 1 annotations. The height is `95 * n + 60` pixels. A test builds a real `GPT` and compares the shapes in the diagram with the shapes returned by `model(ids, return_details=True)`, so the picture cannot drift away from the model.

{{source:viz.training.parameter_breakdown_figure}}

The `total` entry of `GPT.parameter_breakdown()` is excluded from the bars and used in the title; components with zero parameters (positional encoding and a tied output head by default) are not drawn.

## Where each builder is used

| Page | Builders |
|---|---|
| `page_home.py` | `dataflow_figure`, `model_stages` |
| `page_tokenization.py` | `tokens_html`, `token_rows`, `merge_rows`, `vocab_frequency_figure`, `compression_figure` |
| `page_embeddings.py` | `embedding_heatmap`, `embedding_scatter`, `similarity_heatmap` |
| `page_positional.py` | `positional_heatmap`, `positional_curves`, `cosine_similarity_matrix`, `heatmap` |
| `page_attention.py` | `attention_heatmap`, `qkv_heatmap`, `attention_received_figure` |
| `page_multihead.py` | `attention_heatmap`, `heads_grid` |
| `page_norm_ffn.py` | `layernorm_distribution`, `layernorm_token_stats`, `heatmap` |
| `page_transformer.py` | `attention_heatmap`, `heads_grid`, `heatmap`, `model_stages`, `block_stages`, `dataflow_figure`, `parameter_breakdown_figure` |
| `page_inference.py` | `attention_heatmap`, `empty_figure` |
| `page_training.py` | `loss_figure`, `lr_figure`, `grad_norm_figure`, `throughput_figure`, `schedule_figure` |

## Configuration and environment

The layer reads no environment variable and no configuration file. Colours are Plotly defaults plus three fixed values in `dataflow_figure` (`#3b82d4` outline, `#eef4fc` fill, `#57606a` arrows) and a six-colour pastel cycle for token chips. The only runtime parameters are function arguments.

## Runtime behaviour

- Figures are built on the CPU in float64, so the cost is negligible for the default sizes (a 64 x 64 attention matrix, 300 history rows).
- Every chart on a page needs a **unique Streamlit key**. The training page appends a frame counter to its keys because the same chart is redrawn many times while a run streams metrics.
- Colour ranges are fixed for probabilities and similarities, so two figures of the same kind can be compared directly.
- PCA is deterministic up to the sign of each component.

## Tests

The tests check the PCA output shapes and the explained variance, that PCA recovers a dominant axis exactly, the degenerate inputs, the cosine matrix, that masked attention cells are blank (not zero), the layout of the heads grid, the NaN-tolerant series extraction, the moving average, the stage shapes against a real model, and the parameter bars.

## Common pitfalls

- **Two identical keys on one page** raise a Streamlit duplicate-element error. Give every `plot(...)` call its own key.
- **Passing a probability matrix with `-inf` cells** is fine, but passing raw scores to a builder that pins the range to [0, 1] hides the structure; use `pre_softmax=True`.
- **Reading a PCA plot of a random table.** Before training the first two components explain only about 12 % of the variance; there is nothing to see.
- **Expecting merges after loading a checkpoint.** The merge trace is not saved.
